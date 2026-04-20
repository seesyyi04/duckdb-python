#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb_python/python_udf_channel.hpp"
#include "duckdb/main/client_context.hpp"

#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include "duckdb/common/types/arrow_aux_data.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/python_conversion.hpp"

#include <Python.h>
#include <string>
#include <vector>
#include <chrono>

namespace duckdb {

using SteadyClock = std::chrono::steady_clock;
using TimePoint = SteadyClock::time_point;

static double UsElapsed(TimePoint start, TimePoint end) {
	return std::chrono::duration<double, std::micro>(end - start).count();
}

struct LoopProfile {
	double gil_wait_us       = 0;  // blocked waiting for batches (GIL released)
	double batch_collect_us  = 0;  // collecting additional batches from queue
	double arrow_input_us    = 0;  // DuckDB DataChunk → PyArrow columns
	double concat_us         = 0;  // pyarrow.concat_tables (concatenated path only)
	double udf_exec_us      = 0;  // Python UDF execution
	double arrow_output_us   = 0;  // PyArrow result → DuckDB vector (fast path)
	double signal_us         = 0;  // SignalBatchDone overhead
	idx_t  total_batches     = 0;
	idx_t  total_rows        = 0;
	idx_t  total_iterations  = 0;

	void Print() const {
		double total = gil_wait_us + batch_collect_us + arrow_input_us +
		               concat_us + udf_exec_us + arrow_output_us + signal_us;
		if (total == 0) return;

		double busy = total - gil_wait_us;
		double utilization = 100.0 * busy / total;

		fprintf(stderr, "\n===== PythonUDFChannel Profile =====\n");
		fprintf(stderr, "Iterations:       %llu\n", (unsigned long long)total_iterations);
		fprintf(stderr, "Total batches:    %llu\n", (unsigned long long)total_batches);
		fprintf(stderr, "Total rows:       %llu\n", (unsigned long long)total_rows);
		fprintf(stderr, "-----------------------------------\n");
		fprintf(stderr, "GIL wait:         %10.1f us  (%5.1f%%)\n", gil_wait_us,       100.0 * gil_wait_us / total);
		fprintf(stderr, "Batch collect:    %10.1f us  (%5.1f%%)\n", batch_collect_us,  100.0 * batch_collect_us / total);
		fprintf(stderr, "Arrow input:      %10.1f us  (%5.1f%%)\n", arrow_input_us,    100.0 * arrow_input_us / total);
		fprintf(stderr, "Concat tables:    %10.1f us  (%5.1f%%)\n", concat_us,         100.0 * concat_us / total);
		fprintf(stderr, "UDF execution:    %10.1f us  (%5.1f%%)\n", udf_exec_us,       100.0 * udf_exec_us / total);
		fprintf(stderr, "Arrow output:     %10.1f us  (%5.1f%%)\n", arrow_output_us,   100.0 * arrow_output_us / total);
		fprintf(stderr, "Signal:           %10.1f us  (%5.1f%%)\n", signal_us,         100.0 * signal_us / total);
		fprintf(stderr, "-----------------------------------\n");
		fprintf(stderr, "Total accounted:  %10.1f us\n", total);
		fprintf(stderr, "Busy time:        %10.1f us\n", busy);
		fprintf(stderr, "Idle time (wait): %10.1f us\n", gil_wait_us);
		fprintf(stderr, "\n");
		fprintf(stderr, ">>> Python thread utilization: %5.1f%% <<<\n", utilization);
		fprintf(stderr, "    (UDF compute fraction:     %5.1f%%)\n", 100.0 * udf_exec_us / total);
		fprintf(stderr, "    (Conversion overhead:      %5.1f%%)\n",
		        100.0 * (arrow_input_us + concat_us + arrow_output_us) / total);
		fprintf(stderr, "====================================\n\n");
	}
};

static bool ExportToC(py::object &arr, ArrowArray &out) {
	try {
		arr.attr("_export_to_c")(reinterpret_cast<uintptr_t>(&out));
		return true;
	} catch (...) {
		return false;
	}
}

static bool ExportToCWithSchema(py::object &arr, ArrowArray &out_arr, ArrowSchema &out_schema) {
	try {
		arr.attr("_export_to_c")(reinterpret_cast<uintptr_t>(&out_arr), reinterpret_cast<uintptr_t>(&out_schema));
		return true;
	} catch (...) {
		return false;
	}
}

static void ReleaseArrow(ArrowArray &a) {
	if (a.release) { a.release(&a); }
}

static void ReleaseArrow(ArrowSchema &s) {
	if (s.release) { s.release(&s); }
}

// Arrow validity bitmap --> DuckDB validity mask
static void TransferValidity(const ArrowArray &arrow, Vector &out_vec, idx_t row_count) {
	const auto *bitmap = static_cast<const uint8_t *>(arrow.buffers[0]);
	if (!bitmap) return;
 
	idx_t off = static_cast<idx_t>(arrow.offset);
	auto &validity = FlatVector::Validity(out_vec);
	for (idx_t i = 0; i < row_count; i++) {
		idx_t src = off + i;
		if (!((bitmap[src / 8] >> (src % 8)) & 1)) {
			validity.SetInvalid(i);
		}
	}
}

// fast-path writers (BIGINT, DOUBLE, VARCHAR)
static void WriteBigint(const ArrowArray &arrow, Vector &out_vec, idx_t row_count) {
	idx_t off = static_cast<idx_t>(arrow.offset);
	const auto *src = static_cast<const int64_t *>(arrow.buffers[1]) + off;
	memcpy(FlatVector::GetData<int64_t>(out_vec), src, row_count * sizeof(int64_t));
	TransferValidity(arrow, out_vec, row_count);
}

static void WriteDouble(const ArrowArray &arrow, Vector &out_vec, idx_t row_count) {
	idx_t off = static_cast<idx_t>(arrow.offset);
	const auto *src = static_cast<const double *>(arrow.buffers[1]) + off;
	memcpy(FlatVector::GetData<double>(out_vec), src, row_count * sizeof(double));
	TransferValidity(arrow, out_vec, row_count);
}

static void WriteVarchar(const ArrowArray &arrow, const ArrowSchema &schema,
                         Vector &out_vec, idx_t row_count) {
	idx_t off = static_cast<idx_t>(arrow.offset);
	const auto *char_data = static_cast<const char *>(arrow.buffers[2]);
	if (!char_data) { char_data = ""; }
 
	bool large = schema.format && schema.format[0] == 'U';
	if (large) {
		const auto *offsets = static_cast<const int64_t *>(arrow.buffers[1]) + off;
		for (idx_t i = 0; i < row_count; i++) {
			StringVector::AddString(out_vec, char_data + offsets[i],
			                        static_cast<idx_t>(offsets[i + 1] - offsets[i]));
		}
	} else {
		const auto *offsets = static_cast<const int32_t *>(arrow.buffers[1]) + off;
		for (idx_t i = 0; i < row_count; i++) {
			StringVector::AddString(out_vec, char_data + offsets[i],
			                        static_cast<idx_t>(offsets[i + 1] - offsets[i]));
		}
	}
	TransferValidity(arrow, out_vec, row_count);
}

// Pyarrow result --> write into DuckDB vector
static bool WriteResult(py::object &result, Vector &out_vec,
                        idx_t row_count, LogicalTypeId type_id,
                        string &error_message) {
	if (type_id == LogicalTypeId::BIGINT || type_id == LogicalTypeId::DOUBLE) {
		ArrowArray arrow;
		memset(&arrow, 0, sizeof(arrow));
		if (!ExportToC(result, arrow)) {
			error_message = "Failed to export PyArrow array via C Data Interface";
			return false;
		}
		if (type_id == LogicalTypeId::BIGINT) {
			WriteBigint(arrow, out_vec, row_count);
		} else {
			WriteDouble(arrow, out_vec, row_count);
		}
		ReleaseArrow(arrow);
		return true;
	}
 
	if (type_id == LogicalTypeId::VARCHAR) {
		ArrowArray arrow;
		ArrowSchema schema;
		memset(&arrow, 0, sizeof(arrow));
		memset(&schema, 0, sizeof(schema));
		if (!ExportToCWithSchema(result, arrow, schema)) {
			error_message = "Failed to export PyArrow string array via C Data Interface";
			return false;
		}
		WriteVarchar(arrow, schema, out_vec, row_count);
		ReleaseArrow(arrow);
		ReleaseArrow(schema);
		return true;
	}
 
	py::list result_list;
	try {
	    if (py::isinstance<py::list>(result)) {
        	result_list = result;
    	} else {
        	result_list = result.attr("to_pylist")();
    	}
	} catch (...) {
    	error_message = "Could not convert UDF result to list";
    	return false;
	}
	if ((idx_t)py::len(result_list) != row_count) {
    	error_message = "UDF returned wrong number of rows";
    	return false;
	}
	for (idx_t i = 0; i < row_count; i++) {
    	py::handle val = result_list[i];
    	if (val.is_none()) {
        	FlatVector::SetNull(out_vec, i, true);
    	} else {
        	TransformPythonObject(py::reinterpret_borrow<py::object>(val), out_vec, i);
    	}
	}
	return true;
}

static py::object BatchToArrowTable(UDFBatch *batch) {
	DataChunk *input = batch->input;
	auto types = input->GetTypes();
	vector<string> names;
	names.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		names.push_back(StringUtil::Format("c%d", i));
	}
	py::list batch_list;
	TransformDuckToArrowChunk(batch->arrow_schema, batch->arrow_array, batch_list);
	return pyarrow::ToArrowTable(types, names, batch_list, batch->client_props);
}

static py::object CallUDF(PyObject *func, py::tuple &columns,
                          string &error_message, bool *failed) {
	auto ret = PyObject_CallObject(func, columns.ptr());
	if (ret == nullptr && PyErr_Occurred()) {
		*failed = true;
		PyObject *ptype, *pvalue, *ptb;
		PyErr_Fetch(&ptype, &pvalue, &ptb);
		if (pvalue) {
			PyObject *s = PyObject_Str(pvalue);
			if (s) {
				const char *msg = PyUnicode_AsUTF8(s);
				if (msg) error_message = std::string("Python UDF error: ") + msg;
				Py_DECREF(s);
			}
		}
		Py_XDECREF(ptype);
		Py_XDECREF(pvalue);
		Py_XDECREF(ptb);
		return py::none();
	}
	*failed = false;
	return py::reinterpret_steal<py::object>(ret);
}

static py::tuple PrependSelf(py::object &udf_self, py::tuple &cols) {
	idx_t n = py::len(cols);
	py::tuple full_args(n + 1);
	full_args[0] = udf_self;
	for (idx_t i = 0; i < n; i++) {
		full_args[i + 1] = cols[i];
	}
	return full_args;
}

static void SignalBatchDone(UDFBatch *batch) {
	batch->done.store(true, std::memory_order_release);
	auto *waiter = batch->completion_waiters.try_pop();
	if (waiter) { waiter->self_handle.resume(); }
}

PythonUDFChannel::PythonUDFChannel(TaskScheduler &scheduler, std::size_t buffer_capacity) 
	: scheduler(scheduler), database(db) {
		secondary_connection = make_uniq<Connection>(*database);
	}

PythonUDFChannel::~PythonUDFChannel() {
	Stop();
	PyGILState_STATE gstate = PyGILState_Ensure();
	PyGILState_Release(gstate);
}

void PythonUDFChannel::Start() {
	if (running.load()) return;
	running.store(true);
	python_thread = std::make_unique<std::thread>([this]() {
		PythonThreadLoop();
	});
}

void PythonUDFChannel::Stop() {
	if (!running.load()) { return; }
	running.store(false);

	UDFBatch *poison_batch = MakePoison();
	while (!input_buffer.try_push_sync(poison_batch)) {}

	if (python_thread && python_thread->joinable()) {
		if (PyGILState_Check()) {
			PyThreadState *tstate = PyEval_SaveThread();
			python_thread->join();
			PyEval_RestoreThread(tstate);
		} else {
			python_thread->join();
		}
	}
	python_thread.reset();
}

bool PythonUDFChannel::HasError() const { return has_error.load(std::memory_order_acquire); }

std::string PythonUDFChannel::GetError() const {
	if (has_error.load(std::memory_order_acquire)) { return error_message; }
	return {};
}
 
void PythonUDFChannel::PythonThreadLoop() {
	static constexpr int MAX_BATCH_PER_GIL = 4;
	LoopProfile prof;
	PyGILState_STATE gstate = PyGILState_Ensure(); 

	auto Now = []() { return SteadyClock::now(); };

	PyObject *pa_module = PyImport_ImportModule("pyarrow");
	if (!pa_module) {
		has_error.store(true, std::memory_order_release);
		error_message = "Failed to import pyarrow";
		PyErr_Clear();
		PyGILState_Release(gstate);
		return;
	}
	py::object pa = py::reinterpret_borrow<py::object>(pa_module);
	py::object concat = pa.attr("concat_tables");

	while (running.load(std::memory_order_relaxed)) {
		prof.total_iterations++;

		// gil start
		auto t_start = Now();
		PyThreadState *tstate = PyEval_SaveThread();
		auto first = input_buffer.blocking_pop();
		PyEval_RestoreThread(tstate);
		prof.gil_wait_us += UsElapsed(t_start, Now());

        if (!first.has_value()) continue;
        if (IsPoison(*first.value())) break;

		py::dict ns;
		py::exec(R"(
			class UDFContext:
				def __init__(self):
					self.con = None
					self.external_path = None
		)", ns);
		udf_self_object = ns["UDFContext"]();

		// collect batches
		t_start = Now();
		vector<UDFBatch *> group;
		group.reserve(MAX_BATCH_PER_GIL);
		group.push_back(first.value());
		for (int i = 1; i < MAX_BATCH_PER_GIL; i++) {
			auto extra = input_buffer.try_pop();
			if (!extra.has_value()) break;
			if (IsPoison(*extra.value())) break;
			group.push_back(extra.value());
		}
		prof.batch_collect_us += UsElapsed(t_start, Now());
		prof.total_batches += group.size();

		bool should_exit = false;

		if (group[0]->vectorized) {
			try {
				// arrow input 
				fprintf(stderr, "[DEBUG] About to convert input\n");
				t_start = Now();
				py::list tables;
				vector<idx_t> row_counts;
				row_counts.reserve(group.size());
				for (auto *b : group) {
					tables.append(BatchToArrowTable(b));
					row_counts.push_back(b->input->size());
					prof.total_rows += b->input->size();
				}
				prof.arrow_input_us += UsElapsed(t_start, Now());
	
				// concat
				fprintf(stderr, "[DEBUG] About to concat\n");
				t_start = Now();
				py::object merged = concat(tables);
				py::tuple cols(merged.attr("columns"));
				prof.concat_us += UsElapsed(t_start, Now());
	
				// udf execution
				fprintf(stderr, "[DEBUG] About to call UDF\n");
				t_start = Now();
				bool failed = false;
				py::object full_result = CallUDF(group[0]->udf_func, cols, error_message, &failed);
				prof.udf_exec_us += UsElapsed(t_start, Now());
	
				if (failed) {
					has_error.store(true, std::memory_order_release);
					for (auto *b : group) SignalBatchDone(b);
					should_exit = true;
				} else {
					// arrow output
					LogicalTypeId tid = group[0]->return_type.id();
					idx_t offset = 0;
					bool is_list = py::isinstance<py::list>(full_result);
					for (idx_t i = 0; i < group.size(); i++) {
						auto *b = group[i];
						idx_t rc = row_counts[i];
	
						t_start = Now();
						b->result->SetCardinality(rc);
						auto &vec = b->result->data[0];
	 
						py::object slice;
        				if (is_list) {
            				slice = full_result.attr("__getitem__")(py::slice(offset, offset + rc, 1));
        				} else {
        				    slice = full_result.attr("slice")(offset, rc);
        				}

						string err;
						fprintf(stderr, "[DEBUG] WriteResult for type %d, rows=%llu\n", 
        								(int)tid, (unsigned long long)rc);
						if (!WriteResult(slice, vec, rc, tid, err)) {
							fprintf(stderr, "[DEBUG] WriteResult FAILED: %s\n", err.c_str());
							has_error.store(true, std::memory_order_release);
							error_message = err;
							for (idx_t j = i; j < group.size(); j++) SignalBatchDone(group[j]);
							should_exit = true;
							break;
						}
						vec.Flatten(rc);
						prof.arrow_output_us += UsElapsed(t_start, Now());
	
						t_start = Now();
						SignalBatchDone(b);
						fprintf(stderr, "[DEBUG] SignalBatchDone for batch %llu\n", (unsigned long long)i);
						prof.signal_us += UsElapsed(t_start, Now());
						offset += rc;
					}
				}
				fprintf(stderr, "[DEBUG] UDF returned, failed=%d\n", (int)failed);
			} catch (const std::exception &e) {
				fprintf(stderr, "[DEBUG] Exception in arrow path: %s\n", e.what());
				has_error.store(true, std::memory_order_release);
				error_message = std::string("C++ Exception: ") + e.what();
				for (auto *b : group) SignalBatchDone(b);
				should_exit = true;
			} catch (...) {
				has_error.store(true, std::memory_order_release);
				error_message = "Unknown C++ Exception";
				for (auto *b : group) SignalBatchDone(b);
				should_exit = true;
			}
		} else {
			for (auto *batch : group) {
				try {
					DataChunk *input = batch->input;
					DataChunk *result_chunk = batch->result;
					idx_t row_count = input->size();
					result_chunk->SetCardinality(row_count);
					auto &out_vec = result_chunk->data[0];
		
					for (idx_t row = 0; row < row_count; row++) {
						idx_t col_count = input->ColumnCount();
						py::tuple args(col_count);
						bool has_null = false;
						for (idx_t c = 0; c < col_count; c++) {
							auto value = input->data[c].GetValue(row);
							if (value.IsNull()) {
								FlatVector::SetNull(out_vec, row, true);
								has_null = true;
								break;
							}
							args[c] = PythonObject::FromValue(value, input->data[c].GetType(),
													   batch->client_props);
						}
						{
							if (has_null) continue;
							auto ret = py::reinterpret_steal<py::object>(
							PyObject_CallObject(batch->udf_func, args.ptr()));
							if (!ret || ret.is_none()) {
								if (PyErr_Occurred()) { PyErr_Clear(); }
								FlatVector::SetNull(out_vec, row, true);
							} else {
								TransformPythonObject(ret, out_vec, row);
							}
						}
					}
					SignalBatchDone(batch);
				} catch (const std::exception &e) {
            		has_error.store(true, std::memory_order_release);
            		error_message = std::string("C++ Exception: ") + e.what();
            		SignalBatchDone(batch);
            		should_exit = true;
            		break;
        		} catch (...) {
            		has_error.store(true, std::memory_order_release);
            		error_message = "Unknown C++ Exception";
            		SignalBatchDone(batch);
            		should_exit = true;
            		break;
        		}
			}
		}

		if (should_exit) break;
	}
	prof.Print();
	Py_XDECREF(pa_module);
	PyGILState_Release(gstate);
}

PythonUDFTask::PythonUDFTask(PythonUDFChannel &channel, UDFBatch &batch)
	: channel(channel), batch(batch) {}

TaskExecutionResult PythonUDFTask::Execute(TaskExecutionMode mode) {
	return TaskExecutionResult::TASK_ERROR;
}

TaskCoroutine PythonUDFTask::ExecuteAsync(TaskExecutionMode mode) {
	if (channel.HasError()) { 
		co_return TaskExecutionResult::TASK_ERROR;
	}
	// push input batch into the channel for the Python thread
	co_await channel.GetInputBuffer().push(&batch);
	
	co_await DoneAwaitable{batch.done, batch.completion_waiters};

	if (channel.HasError()) {
		co_return TaskExecutionResult::TASK_ERROR;
	}
	co_return TaskExecutionResult::TASK_FINISHED;
}

} // namespace duckdb
