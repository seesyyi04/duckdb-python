#pragma once

#include "duckdb/parallel/task.hpp"
#include "duckdb/parallel/task_coroutine.hpp"
#include "duckdb/parallel/ring_buffer.hpp"

#include <Python.h>
#include <atomic>
#include <thread>
#include <memory>

struct ArrowArray;
struct ArrowSchema;

namespace duckdb {

class TaskScheduler;

// Payload moved through the ring buffers.
// Wraps the Arrow C Data Interface structs that represent a batch of data.
struct ArrowBatch {
	ArrowArray *array = nullptr;
	ArrowSchema *schema = nullptr;
};

// when C++ is done sending data, push a poison pill
inline bool IsPoison(const ArrowBatch &batch) { return batch.array == nullptr; }
inline ArrowBatch MakePoison() { return ArrowBatch{nullptr, nullptr}; }

static constexpr std::size_t DEFAULT_CHANNEL_CAPACITY = 64;

class PythonUDFChannel {
	public:
		PythonUDFChannel(PyObject *udf_func, TaskScheduler &scheduler, std::size_t buffer_capacity = DEFAULT_CHANNEL_CAPACITY);
		~PythonUDFChannel();
		
		// Not copyabble or movable - Python thread holds references to internals
		PythonUDFChannel(const PythonUDFChannel &) = delete;
		PythonUDFChannel &operator=(const PythonUDFChannel &) = delete;

		// start python executor thread
		void Start();
		// signal shutdown, join the python thread
		void Stop();
		bool HasError() const;
		std::string GetError() const;
		// access the buffers
		MPSCRingBuffer<ArrowBatch, DEFAULT_CHANNEL_CAPACITY> &GetInputBuffer() { return input_buffer; }
		SPMCRingBuffer<ArrowBatch, DEFAULT_CHANNEL_CAPACITY> &GetOutputBuffer() { return output_buffer; }
	private:
		void PythonThreadLoop();

		// python callable
		PyObject *udf_func;

		TaskScheduler &scheduler;

		// C++ workers push input here and Python pops
		MPSCRingBuffer<ArrowBatch, DEFAULT_CHANNEL_CAPACITY> input_buffer;
		// vice versa
		SPMCRingBuffer<ArrowBatch, DEFAULT_CHANNEL_CAPACITY> output_buffer;

		std::unique_ptr<std::thread> python_thread;
		std::atomic<bool> running{false};
		std::atomic<bool> has_error{false};
		std::string error_message;
};

// task that participates in DuckDB's parallel execution
class PythonUDFTask : public Task {
	public:
		PythonUDFTask(PythonUDFChannel &channel, ArrowBatch input);
		TaskExecutionResult Execute(TaskExecutionMode mode) override;
		TaskCoroutine ExecuteAsync(TaskExecutionMode mode) override;
		ArrowBatch &GetResult() { return result; }
	private:
		PythonUDFChannel& channel;
		ArrowBatch input, result;
};

} // namespace duckdb