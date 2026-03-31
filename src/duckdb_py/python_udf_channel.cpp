#include "duckdb_python/python_udf_channel.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

#include <Python.h>
#include <stdexcept>
#include <string>

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

struct ArrowSchema {
	const char *format, *name, *metadata;
	int64_t flags, n_children;
	struct ArrowSchema **children, *dictionary;
	void (*release)(struct ArrowSchema *);
	void *private_data;
};

struct ArrowArray {
	int64_t length, null_count, offset, n_buffers, n_children;
	const void **buffers;
	struct ArrowArray **children, *dictionary;
	void (*release)(struct ArrowArray *);
	void *private_data;
};
#endif

namespace duckdb {
	PythonUDFChannel::PythonUDFChannel(PyObject *udf_func, TaskScheduler &scheduler, std::size_t buffer_capacity) 
		: udf_func(udf_func), scheduler(scheduler) { Py_INCREF(udf_func); }
	PythonUDFChannel::~PythonUDFChannel() {
		Stop();
		PyGILState_STATE gstate = PyGILState_Ensure();
		Py_DECREF(udf_func);
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

		ArrowBatch poison = MakePoison();
		while (!input_buffer.try_push_sync(poison)) {
			// spin
		}
		if (python_thread && python_thread->joinable()) { python_thread->join(); }
		python_thread.reset();
	}

	bool PythonUDFChannel::HasError() const { return has_error.load(std::memory_order_acquire); }

	std::string PythonUDFChannel::GetError() const {
		if (has_error.load(std::memory_order_acquire)) { return error_message; }
		return {};
	}

	void PythonUDFChannel::PythonThreadLoop() {
		// acquire GIL for this thread
		PyGILState_STATE gstate = PyGILState_Ensure();
		// import pyarrow
		PyObject *pa_module = PyImport_ImportModule("pyarrow");
		if (!pa_module) {
			has_error.store(true, std::memory_order_release);
			error_message = "Failed to import pyarrow";
			PyErr_Clear();
			PyGILState_Release(gstate);
			return;
		}

		while (true) {
			// release GIL while waiting for C++ input
			Py_BEGIN_ALLOW_THREADS

			// busy-poll the MPSC buffer; single consumer path, no contention
			std::optional<ArrowBatch> maybe_batch;
			while (running.load(std::memory_order_relaxed)) {
				maybe_batch = input_buffer.try_pop();
				if (maybe_batch.has_value()) break;
				// yield to avoid burning CPU while waiting
				std::this_thread::yield();
			}

			Py_END_ALLOW_THREADS

			if (!maybe_batch.has_value()) break; 
			ArrowBatch batch = std::move(maybe_batch.value());

			if (IsPoison(batch)) break;

			PyObject *pa_record_batch_type = PyObject_GetAttrString(pa_module, "RecordBatch");
			if (!pa_record_batch_type) {
				has_error.store(true, std::memory_order_release);
				error_message = "Failed to get pyarrow.RecordBatch";
				PyErr_Clear();
				break;
			}

			PyObject *import_func = PyObject_GetAttrString(pa_record_batch_type, "_import_from_c");
			Py_DECREF(pa_record_batch_type);
			if (!import_func) {
				has_error.store(true, std::memory_order_release);
				error_message = "Failed to get RecordBatch._import_from_c";
				PyErr_Clear();
				break;
			}

			PyObject *args = Py_BuildValue("(nn)",
										  reinterpret_cast<Py_ssize_t>(batch.array),
										  reinterpret_cast<Py_ssize_t>(batch.schema));
			PyObject *py_batch = PyObject_CallObject(import_func, args);
			Py_DECREF(import_func);
			Py_DECREF(args);

			if (!py_batch) {
				has_error.store(true, std::memory_order_release);
				error_message = "Failed to import Arrow batch into PyArrow";
				PyErr_Clear();
				break;
			}

			// call the UDF
			PyObject *call_args = PyTuple_Pack(1, py_batch);
			PyObject *py_result = PyObject_CallObject(udf_func, call_args);
			Py_DECREF(call_args);
			Py_DECREF(py_batch);

			if (!py_result) {
				has_error.store(true, std::memory_order_release);
				// extract python exception message
				PyObject *ptype, *pvalue, *ptraceback;
				PyErr_Fetch(&ptype, &pvalue, &ptraceback);
				if (pvalue) {
					PyObject *str_obj = PyObject_Str(pvalue);
					if (str_obj) {
						const char *err_str = PyUnicode_AsUTF8(str_obj);
						if (err_str) {
							error_message = std::string("Python UDF error: ") + err_str;
						}
						Py_DECREF(str_obj);
					}
				}
				Py_XDECREF(ptype);
				Py_XDECREF(pvalue);
				Py_XDECREF(ptraceback);
				break;
			}
			// export result back to Arrow C data interface
			ArrowArray *result_array = new ArrowArray();
			ArrowSchema *result_schema = new ArrowSchema();
			memset(result_array, 0, sizeof(ArrowArray));
			memset(result_schema, 0, sizeof(ArrowSchema));
	
			PyObject *export_func = PyObject_GetAttrString(py_result, "_export_to_c");
			if (!export_func) {
				has_error.store(true, std::memory_order_release);
				error_message = "Result does not support _export_to_c";
				PyErr_Clear();
				Py_DECREF(py_result);
				delete result_array;
				delete result_schema;
				break;
			}
	
			PyObject *export_args = Py_BuildValue("(nn)",
												  reinterpret_cast<Py_ssize_t>(result_array),
												  reinterpret_cast<Py_ssize_t>(result_schema));
			PyObject *export_ret = PyObject_CallObject(export_func, export_args);
			Py_DECREF(export_func);
			Py_DECREF(export_args);
			Py_DECREF(py_result);
	
			if (!export_ret) {
				has_error.store(true, std::memory_order_release);
				error_message = "Failed to export result to Arrow C Data Interface";
				PyErr_Clear();
				delete result_array;
				delete result_schema;
				break;
			}
			Py_DECREF(export_ret);
	
			// push result to SPMC buffer for C++ consumers
			ArrowBatch result_batch{result_array, result_schema};
	
			// release GIL while pushing so C++ coroutines can resume
			Py_BEGIN_ALLOW_THREADS
			while (!output_buffer.try_push(result_batch)) std::this_thread::yield();
			Py_END_ALLOW_THREADS
		}
		Py_DECREF(pa_module);
		PyGILState_Release(gstate);
	}


PythonUDFTask::PythonUDFTask(PythonUDFChannel &channel, ArrowBatch input)
	: channel(channel), input(std::move(input)) {}

TaskExecutionResult PythonUDFTask::Execute(TaskExecutionMode mode) {
	return TaskExecutionResult::TASK_ERROR;
}

TaskCoroutine PythonUDFTask::ExecuteAsync(TaskExecutionMode mode) {
	if (channel.HasError()) { 
		co_return TaskExecutionResult::TASK_ERROR;
	}
	// push input batch into the channel for the Python thread
	co_await channel.GetInputBuffer().push(input);
	
	// wait for a result to come back from the python thread
	co_await channel.GetOutputBuffer().pop(result);

	if (channel.HasError()) {
		co_return TaskExecutionResult::TASK_ERROR;
	}
	co_return TaskExecutionResult::TASK_FINISHED;
}

} // namespace duckdb
