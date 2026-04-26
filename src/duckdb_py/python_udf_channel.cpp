#include "duckdb/parallel/task_scheduler.hpp"
#include "duckdb_python/python_udf_channel.hpp"
#include "duckdb/main/client_context.hpp"
#include "duckdb_python/python_conversion.hpp"
 
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include "duckdb/common/types/arrow_aux_data.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"

#include <Python.h>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>
#include <cstdio>

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

		fprintf(stdout, "\n===== PythonUDFChannel Profile =====\n");
		fprintf(stdout, "Iterations:       %llu\n", (unsigned long long)total_iterations);
		fprintf(stdout, "Total batches:    %llu\n", (unsigned long long)total_batches);
		fprintf(stdout, "Total rows:       %llu\n", (unsigned long long)total_rows);
		fprintf(stdout, "-----------------------------------\n");
		fprintf(stdout, "GIL wait:         %10.1f us  (%5.1f%%)\n", gil_wait_us,       100.0 * gil_wait_us / total);
		fprintf(stdout, "Batch collect:    %10.1f us  (%5.1f%%)\n", batch_collect_us,  100.0 * batch_collect_us / total);
		fprintf(stdout, "Arrow input:      %10.1f us  (%5.1f%%)\n", arrow_input_us,    100.0 * arrow_input_us / total);
		fprintf(stdout, "Concat tables:    %10.1f us  (%5.1f%%)\n", concat_us,         100.0 * concat_us / total);
		fprintf(stdout, "UDF execution:    %10.1f us  (%5.1f%%)\n", udf_exec_us,       100.0 * udf_exec_us / total);
		fprintf(stdout, "Arrow output:     %10.1f us  (%5.1f%%)\n", arrow_output_us,   100.0 * arrow_output_us / total);
		fprintf(stdout, "Signal:           %10.1f us  (%5.1f%%)\n", signal_us,         100.0 * signal_us / total);
		fprintf(stdout, "-----------------------------------\n");
		fprintf(stdout, "Total accounted:  %10.1f us\n", total);
		fprintf(stdout, "Busy time:        %10.1f us\n", busy);
		fprintf(stdout, "Idle time (wait): %10.1f us\n", gil_wait_us);
		fprintf(stdout, "\n");
		fprintf(stdout, ">>> Python thread utilization: %5.1f%% <<<\n", utilization);
		fprintf(stdout, "    (UDF compute fraction:     %5.1f%%)\n", 100.0 * udf_exec_us / total);
		fprintf(stdout, "    (Conversion overhead:      %5.1f%%)\n",
		        100.0 * (arrow_input_us + concat_us + arrow_output_us) / total);
		fprintf(stdout, "====================================\n\n");
		fflush(stdout);
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

static bool WriteArrowToVector(const ArrowArray &arrow, const ArrowSchema &schema, Vector &out_vec, idx_t row_count, idx_t extra_offset = 0);

// Arrow validity bitmap --> DuckDB validity mask
static void validity(const ArrowArray &arrow, Vector &out_vec, idx_t row_count, idx_t offset) {
	const auto *bitmap = static_cast<const uint8_t *>(arrow.buffers[0]);
	if (!bitmap) return;
	auto &validity = FlatVector::Validity(out_vec);
	for (idx_t i = 0; i < row_count; i++) {
		idx_t src = offset + i;
		if (!((bitmap[src / 8] >> (src % 8)) & 1)) {
			validity.SetInvalid(i);
		}
	}
}

static bool WriteArrowToVector(const ArrowArray &arrow, const ArrowSchema &schema, Vector &out_vec, idx_t row_count, idx_t extra_offset) {
	if (!schema.format) return false;
	idx_t off = static_cast<idx_t>(arrow.offset) + extra_offset;
	const char *fmt = schema.format;

	if (fmt[1] == '\0') {
		switch (fmt[0]) {
			case 'c': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<int8_t>(out_vec),
					   static_cast<const int8_t *>(arrow.buffers[1]) + off,
					   row_count * sizeof(int8_t));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			case 's': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<int16_t>(out_vec),
					   static_cast<const int16_t *>(arrow.buffers[1]) + off,
					   row_count * sizeof(int16_t));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			case 'i': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<int32_t>(out_vec),
					   static_cast<const int32_t *>(arrow.buffers[1]) + off,
					   row_count * sizeof(int32_t));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			case 'l': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<int64_t>(out_vec),
					   static_cast<const int64_t *>(arrow.buffers[1]) + off,
					   row_count * sizeof(int64_t));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			case 'C': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<uint8_t>(out_vec),
					   static_cast<const uint8_t *>(arrow.buffers[1]) + off,
					   row_count * sizeof(uint8_t));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			case 'S': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<uint16_t>(out_vec),
					   static_cast<const uint16_t *>(arrow.buffers[1]) + off,
					   row_count * sizeof(uint16_t));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			case 'I': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<uint32_t>(out_vec),
					   static_cast<const uint32_t *>(arrow.buffers[1]) + off,
					   row_count * sizeof(uint32_t));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			case 'L': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<uint64_t>(out_vec),
					   static_cast<const uint64_t *>(arrow.buffers[1]) + off,
					   row_count * sizeof(uint64_t));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			case 'f': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<float>(out_vec),
					   static_cast<const float *>(arrow.buffers[1]) + off,
					   row_count * sizeof(float));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			case 'g': {
				if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
				memcpy(FlatVector::GetData<double>(out_vec),
					   static_cast<const double *>(arrow.buffers[1]) + off,
					   row_count * sizeof(double));
				validity(arrow, out_vec, row_count, off);
				return true;
			}
			default:
				break;
		}
	}

	if ((fmt[0] == 'u' || fmt[0] == 'U') && fmt[1] == '\0') {
		if (arrow.n_buffers < 3) return false;
		const auto *char_data = static_cast<const char *>(arrow.buffers[2]);
		if (!char_data) char_data = "";

		if (fmt[0] == 'U') {
			const auto *offsets = static_cast<const int64_t *>(arrow.buffers[1]) + off;
			for (idx_t i = 0; i < row_count; i++) {
				StringVector::AddString(out_vec, char_data + offsets[i], static_cast<idx_t>(offsets[i + 1] - offsets[i]));
			}
		} else {
			const auto *offsets = static_cast<const int32_t *>(arrow.buffers[1]) + off;
			for (idx_t i = 0; i < row_count; i++) {
				StringVector::AddString(out_vec, char_data + offsets[i], static_cast<idx_t>(offsets[i + 1] - offsets[i]));
			}
		}
		validity(arrow, out_vec, row_count, off);
		return true;
	}

	if (fmt[0] == '+' && fmt[1] == 's') {
		if (arrow.n_children < 1) return false;
		auto &child_vectors = StructVector::GetEntries(out_vec);
		if ((idx_t)arrow.n_children != child_vectors.size()) return false;

		for (idx_t c = 0; c < (idx_t)arrow.n_children; c++) {
			if (!arrow.children[c] || !schema.children[c]) return false;
			if (!WriteArrowToVector(*arrow.children[c], *schema.children[c], *child_vectors[c], row_count, extra_offset)) {
				return false;
			}
		}
		validity(arrow, out_vec, row_count, off);
		return true;
	}

	if (fmt[0] == '+' && (fmt[1] == 'l' || fmt[1] == 'L')) {
		if (arrow.n_children < 1 || !arrow.children[0] || !schema.children[0]) return false;
		if (arrow.n_buffers < 2 || !arrow.buffers[1]) return false;
		auto &child_vec = ListVector::GetEntry(out_vec);
		auto list_data = FlatVector::GetData<list_entry_t>(out_vec);
		idx_t total_child_count = 0;

		if (fmt[1] == 'L') {
			const auto *offsets = static_cast<const int64_t *>(arrow.buffers[1]) + off;
			for (idx_t i = 0; i < row_count; i++) {
				idx_t start = static_cast<idx_t>(offsets[i]);
				idx_t len = static_cast<idx_t>(offsets[i + 1] - offsets[i]);
				list_data[i].offset = start;
				list_data[i].length = len;
				total_child_count += len;
			}
		} else {
			const auto *offsets = static_cast<const int32_t *>(arrow.buffers[1]) + off;
			for (idx_t i = 0; i < row_count; i++) {
				idx_t start = static_cast<idx_t>(offsets[i]);
				idx_t len = static_cast<idx_t>(offsets[i + 1] - offsets[i]);
				list_data[i].offset = start;
				list_data[i].length = len;
				total_child_count += len;
			}
		}
		ListVector::SetListSize(out_vec, total_child_count);
		ListVector::Reserve(out_vec, total_child_count);
		if (total_child_count > 0) {
			if (!WriteArrowToVector(*arrow.children[0], *schema.children[0], child_vec, total_child_count, extra_offset)) {
				return false;
			}
		}
		validity(arrow, out_vec, row_count, off);
		return true;
	}
	return false;
}

// Pyarrow result --> write into DuckDB vector
static bool WriteResult(py::object &result, Vector &out_vec, idx_t row_count, LogicalTypeId type_id, string &error_message) {
 
	if (!py::isinstance<py::list>(result)) {
		ArrowArray arrow;
		ArrowSchema schema;
		memset(&arrow, 0, sizeof(arrow));
		memset(&schema, 0, sizeof(schema));
		if (ExportToCWithSchema(result, arrow, schema)) {
			bool ok = WriteArrowToVector(arrow, schema, out_vec, row_count);
			ReleaseArrow(arrow);
			ReleaseArrow(schema);
			if (ok) return true;
		}
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
	batch->done_cv.notify_all();
}

PythonUDFChannel::PythonUDFChannel(TaskScheduler &scheduler, shared_ptr<DuckDB> db,
                                   const std::string &db_path, std::size_t buffer_capacity) 
	: scheduler(scheduler), database(db), database_path(db_path) {
	}

PythonUDFChannel::~PythonUDFChannel() {
	Stop();
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
	static constexpr int MAX_BATCH_PER_GIL = 32;
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

	py::object udf_self;
	try {
		py::object types_mod = py::module::import("types");
    	udf_self = types_mod.attr("SimpleNamespace")();

		auto py_conn = make_shared_ptr<DuckDBPyConnection>();
		{
			py::gil_scoped_release release;
			py_conn->con.SetDatabase(database);
			py_conn->con.SetConnection(make_uniq<Connection>(py_conn->con.GetDatabase()));
		}
		udf_self.attr("con") = py::cast(py_conn);
		udf_self.attr("external_path") = "";
	} catch (const std::exception &e) {
		fprintf(stderr, "[WARN] Could not create UDF self object: %s\n", e.what());
		udf_self = py::none();
	} catch (...) {
		fprintf(stderr, "[WARN] Could not create UDF self object\n");
		udf_self = py::none();
	}

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

		has_error.store(false, std::memory_order_release);
		error_message.clear();

		bool should_exit = false;

		if (!udf_self.is_none() && group[0]->need_self && !group[0]->external_path.empty()) {
			try {
				udf_self.attr("external_path") = group[0]->external_path;
			} catch (...) {}
		}

		if (group[0]->vectorized) {
			try {
				// arrow input 
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
				t_start = Now();
				py::object merged = concat(tables);
				py::tuple cols(merged.attr("columns"));
				prof.concat_us += UsElapsed(t_start, Now());
	
				// udf execution
				t_start = Now();
				bool failed = false;
				py::tuple call_args = cols;
				if (group[0]->need_self && !udf_self.is_none()) {
					call_args = PrependSelf(udf_self, cols);
				}
				py::object full_result = CallUDF(group[0]->udf_func, call_args, error_message, &failed);
				prof.udf_exec_us += UsElapsed(t_start, Now());
	
				if (failed) {
					has_error.store(true, std::memory_order_release);
					for (auto *b : group) SignalBatchDone(b);
					should_exit = true;
				} else {
					// arrow output
					if (py::isinstance<py::list>(full_result)) {
						try {
							LogicalTypeId tid = group[0]->return_type.id();
							py::object pa_type = py::none();
							if (tid == LogicalTypeId::STRUCT) {
								auto &children = StructType::GetChildTypes(group[0]->return_type);
								py::list fields;
								for (auto &child : children) {
									py::object field_type;
									switch (child.second.id()) {
										case LogicalTypeId::DOUBLE:  field_type = pa.attr("float64")(); break;
										case LogicalTypeId::BIGINT:  field_type = pa.attr("int64")(); break;
										case LogicalTypeId::INTEGER: field_type = pa.attr("int32")(); break;
										case LogicalTypeId::VARCHAR: field_type = pa.attr("utf8")(); break;
										default: throw std::runtime_error("unsupported child type");
									}
									fields.append(pa.attr("field")(child.first, field_type));
								}
								pa_type = pa.attr("struct")(fields);
							} else if (tid == LogicalTypeId::BIGINT) {
								pa_type = pa.attr("int64")();
							} else if (tid == LogicalTypeId::DOUBLE) {
								pa_type = pa.attr("float64")();
							} else if (tid == LogicalTypeId::VARCHAR) {
								pa_type = pa.attr("utf8")();
							}
							if (!pa_type.is_none()) {
								full_result = pa.attr("array")(full_result, py::arg("type") = pa_type);
							}
						} catch (...) {} 
					}

					auto t_out_start = Now();
					LogicalTypeId tid = group[0]->return_type.id();
					bool is_list_result = py::isinstance<py::list>(full_result);

					if (!is_list_result) {
						ArrowArray full_arrow;
						ArrowSchema full_schema;
						memset(&full_arrow, 0, sizeof(full_arrow));
						memset(&full_schema, 0, sizeof(full_schema));

						bool exported = ExportToCWithSchema(full_result, full_arrow, full_schema);
						if (exported) {
							idx_t batch_offset = 0;
							for (idx_t i = 0; i < group.size(); i++) {
								auto *b = group[i];
								idx_t rc = row_counts[i];
								b->result->SetCardinality(rc);
								auto &vec = b->result->data[0];

								bool ok = WriteArrowToVector(full_arrow, full_schema, vec, rc, batch_offset);
								vec.Flatten(rc);

								if (!ok) {
									has_error.store(true, std::memory_order_release);
									error_message = "Failed to write Arrow result to vector";
									for (idx_t j = i; j < group.size(); j++) SignalBatchDone(group[j]);
									should_exit = true;
									break;
								}
								SignalBatchDone(b);
								batch_offset += rc;
							}
							ReleaseArrow(full_arrow);
							ReleaseArrow(full_schema);
						} else {
							idx_t offset = 0;
							for (idx_t i = 0; i < group.size(); i++) {
								auto *b = group[i];
								idx_t rc = row_counts[i];
								b->result->SetCardinality(rc);
								auto &vec = b->result->data[0];

								py::object slice = full_result.attr("slice")(offset, rc);
								string err;
								if (!WriteResult(slice, vec, rc, tid, err)) {
									has_error.store(true, std::memory_order_release);
									error_message = err;
									for (idx_t j = i; j < group.size(); j++) SignalBatchDone(group[j]);
									should_exit = true;
									break;
								}
								vec.Flatten(rc);
								SignalBatchDone(b);
								offset += rc;
							}
						}
					} else {
						idx_t offset = 0;
						for (idx_t i = 0; i < group.size(); i++) {
							auto *b = group[i];
							idx_t rc = row_counts[i];
							b->result->SetCardinality(rc);
							auto &vec = b->result->data[0];

							py::object slice = full_result.attr("__getitem__")(py::slice(offset, offset + rc, 1));
							string err;
							if (!WriteResult(slice, vec, rc, tid, err)) {
								has_error.store(true, std::memory_order_release);
								error_message = err;
								for (idx_t j = i; j < group.size(); j++) SignalBatchDone(group[j]);
								should_exit = true;
								break;
							}
							vec.Flatten(rc);
							SignalBatchDone(b);
							offset += rc;
						}
					}
					prof.arrow_output_us += UsElapsed(t_out_start, Now());
				}
			} catch (const std::exception &e) {
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
					prof.total_rows += row_count;
		
					t_start = Now();
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
						if (has_null) continue;
						py::tuple call_args = args;
						if (batch->need_self && !udf_self.is_none()) {
							py::tuple full_args(col_count + 1);
							full_args[0] = udf_self;
							for (idx_t c = 0; c < col_count; c++) {
								full_args[c + 1] = args[c];
							}
							call_args = full_args;
						}

						auto ret = py::reinterpret_steal<py::object>(PyObject_CallObject(batch->udf_func, args.ptr()));
						if (!ret || ret.is_none()) {
							if (PyErr_Occurred()) { PyErr_Clear(); }
							FlatVector::SetNull(out_vec, row, true);
						} else {
							TransformPythonObject(ret, out_vec, row);
						}
					}
					prof.udf_exec_us += UsElapsed(t_start, Now());
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

	if (!udf_self.is_none()) {
		try {
			udf_self.attr("con") = py::none();
		} catch (...) {}
	}
	udf_self = py::none();

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
