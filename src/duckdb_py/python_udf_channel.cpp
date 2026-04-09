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

#include <Python.h>
#include <string>
#include <vector>
#include <chrono>

namespace duckdb {
PythonUDFChannel::PythonUDFChannel(TaskScheduler &scheduler, std::size_t buffer_capacity) 
	: scheduler(scheduler) { }
PythonUDFChannel::~PythonUDFChannel() {
	Stop();
	PyGILState_STATE gstate = PyGILState_Ensure();
	PyGILState_Release(gstate);
}

void PythonUDFChannel::Start() {
	if (running.load()) { return; }
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

	if (python_thread && python_thread->joinable()) python_thread->join();
	python_thread.reset();
}

bool PythonUDFChannel::HasError() const { return has_error.load(std::memory_order_acquire); }

std::string PythonUDFChannel::GetError() const {
	if (has_error.load(std::memory_order_acquire)) { return error_message; }
	return {};
}

static void SignalBatchDone(UDFBatch *batch) {
	batch->done.store(true, std::memory_order_release);
	auto *waiter = batch->completion_waiters.try_pop();
	if (waiter) {
		waiter->self_handle.resume();
	}
}
 
void PythonUDFChannel::PythonThreadLoop() {
	static constexpr int MAX_BATCH_PER_GIL = 4;
	PyGILState_STATE gstate = PyGILState_Ensure(); 

	PyObject *pa_module = PyImport_ImportModule("pyarrow");
	if (!pa_module) {
		has_error.store(true, std::memory_order_release);
		error_message = "Failed to import pyarrow";
		PyErr_Clear();
		PyGILState_Release(gstate);
		return;
	}

	while (running.load(std::memory_order_relaxed)) {
		PyThreadState *tstate = PyEval_SaveThread();
		auto first = input_buffer.blocking_pop();
		PyEval_RestoreThread(tstate);

        if (!first.has_value()) continue;
        if (IsPoison(*first.value())) break;

		// ---- COLLECT BATCHES ----
		vector<UDFBatch *> batch_group;
		batch_group.reserve(MAX_BATCH_PER_GIL);
		batch_group.push_back(first.value());
		for (int i = 1; i < MAX_BATCH_PER_GIL; i++) {
			auto extra = input_buffer.try_pop();
			if (!extra.has_value()) break;
			if (IsPoison(*extra.value())) break;
			batch_group.push_back(extra.value());
		}
 
		// ---- PROCESS ALL BATCHES WITH GIL HELD ----
		bool should_exit = false;
 
		for (auto *batch : batch_group) {
			try {
				DataChunk *input_chunk = batch->input;
				DataChunk *result_chunk = batch->result;
			
				// ---- CONVERT DataChunk → PyArrow Table ----	
				auto types = input_chunk->GetTypes();
				vector<string> names;
				names.reserve(types.size());
				for (idx_t i = 0; i < types.size(); i++) {
					names.push_back(StringUtil::Format("c%d", i));
				}
	 
				py::list single_batch_list;
				TransformDuckToArrowChunk(batch->arrow_schema, batch->arrow_array, single_batch_list);
	 
				py::object pyarrow_table = pyarrow::ToArrowTable(types, names, single_batch_list, batch->client_props);
				py::tuple column_list = pyarrow_table.attr("columns");
				idx_t row_count = input_chunk->size();
				auto t1 = std::chrono::high_resolution_clock::now();
	 
				// ---- CALL UDF ----
				auto ret = PyObject_CallObject(batch->udf_func, column_list.ptr());
	
				if (ret == nullptr && PyErr_Occurred()) {
					has_error.store(true, std::memory_order_release);
					PyObject *ptype, *pvalue, *ptraceback;
					PyErr_Fetch(&ptype, &pvalue, &ptraceback);
					if (pvalue) {
						PyObject *str_obj = PyObject_Str(pvalue);
						if (str_obj) {
							const char *err_str = PyUnicode_AsUTF8(str_obj);
							if (err_str) error_message = std::string("Python UDF error: ") + err_str;
							Py_DECREF(str_obj);
						}
					}
					Py_XDECREF(ptype);
					Py_XDECREF(pvalue);
					Py_XDECREF(ptraceback);
					SignalBatchDone(batch);
					should_exit = true;
					break;
				}
				py::object python_result = py::reinterpret_steal<py::object>(ret);
				
				// Convert result to Python list
				py::list result_list;
				try {
					if (py::isinstance<py::list>(python_result)) {
						result_list = python_result;
					} else {
						result_list = python_result.attr("to_pylist")();
					}
				} catch (...) {
					has_error.store(true, std::memory_order_release);
					error_message = "Could not convert UDF result to list";
					SignalBatchDone(batch);
					should_exit = true;
					break;
				}
	
				if ((idx_t)py::len(result_list) != row_count) {
					has_error.store(true, std::memory_order_release);
					error_message = StringUtil::Format("UDF returned %d rows, expected %d",
													   (int)py::len(result_list), row_count);
					SignalBatchDone(batch);
					should_exit = true;
					break;
				}

				result_chunk->SetCardinality(row_count);
				
				auto &out_vec = result_chunk->data[0];
				for (idx_t i = 0; i < row_count; i++) {
					py::handle val = result_list[i];
					if (val.is_none()) {
						FlatVector::SetNull(out_vec, i, true);
					} else {
						TransformPythonObject(py::reinterpret_borrow<py::object>(val), out_vec, i);
					}
				}
				out_vec.Flatten(row_count);
				result_list = py::list();   // release list
				python_result = py::none(); // release UDF return value
				column_list = py::tuple();  // release input columns
				pyarrow_table = py::none(); // release arrow table
	 
				// ---- SIGNAL THIS BATCH DONE ----
				SignalBatchDone(batch);

			} catch (const std::exception& e) {
                // Catches DuckDB InternalExceptions or pybind11 errors cleanly
                has_error.store(true, std::memory_order_release);
                error_message = std::string("C++ Exception in Python thread: ") + e.what();
                SignalBatchDone(batch);
                should_exit = true;
                break;
            } catch (...) {
                has_error.store(true, std::memory_order_release);
                error_message = "Unknown C++ Exception in Python thread";
                SignalBatchDone(batch);
                should_exit = true;
                break;
            }
		}
		if (should_exit) break;
	}
	Py_DECREF(pa_module);
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
