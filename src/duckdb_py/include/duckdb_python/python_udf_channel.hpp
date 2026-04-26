#pragma once

#include "duckdb/parallel/task.hpp"
#include "duckdb/parallel/task_coroutine.hpp"
#include "duckdb/parallel/ring_buffer.hpp"
#include "duckdb/common/types/data_chunk.hpp"
#include "duckdb/main/client_properties.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/main/connection.hpp"
#include "duckdb/main/database.hpp"

#include <Python.h>
#include <atomic>
#include <thread>
#include <memory>
#include <string>
#include <mutex>
#include <condition_variable>

namespace duckdb {

class TaskScheduler;
class ClientContext;

struct UDFBatch {
	DataChunk *input = nullptr;
	ClientContext *context = nullptr;
	DataChunk *result = nullptr;
	PyObject *udf_func = nullptr;
	LogicalType return_type; 
	std::atomic<bool> done{false};
	WaiterStack completion_waiters;
	std::mutex done_mutex;
	std::condition_variable done_cv;
	bool poison = false;
	bool vectorized = true; // vectorized, not native
	bool need_self = false; // default no self, mostly for scalar
	std::string external_path;
	ArrowSchema arrow_schema;
    ArrowArray arrow_array;
    ClientProperties client_props;
};

inline bool IsPoison(const UDFBatch &batch) { return batch.poison; }
inline UDFBatch *MakePoison() {
	static UDFBatch b;
	b.poison = true;
	return &b;
}

static constexpr std::size_t DEFAULT_CHANNEL_CAPACITY = 64;

class PythonUDFChannel {
	public:
		PythonUDFChannel(TaskScheduler &scheduler, shared_ptr<DuckDB> db, 
						 const std::string &db_path, std::size_t buffer_capacity = DEFAULT_CHANNEL_CAPACITY);
		~PythonUDFChannel();
		
		PythonUDFChannel(const PythonUDFChannel &) = delete;
		PythonUDFChannel &operator=(const PythonUDFChannel &) = delete;

		void Start();
		void Stop();
		bool HasError() const;
		std::string GetError() const;
		MPSCRingBuffer<UDFBatch *, DEFAULT_CHANNEL_CAPACITY> &GetInputBuffer() { return input_buffer; }

	private:
		void PythonThreadLoop();

		TaskScheduler &scheduler;
		shared_ptr<DuckDB> database;
		std::string database_path;

		// C++ workers push input here and Python pops
		MPSCRingBuffer<UDFBatch *, DEFAULT_CHANNEL_CAPACITY> input_buffer;

		std::unique_ptr<std::thread> python_thread;
		std::atomic<bool> running{false};
		std::atomic<bool> has_error{false};
		std::string error_message;
};

// task that participates in DuckDB's parallel execution
class PythonUDFTask : public Task {
	public:
		PythonUDFTask(PythonUDFChannel &channel, UDFBatch &batch);
		TaskExecutionResult Execute(TaskExecutionMode mode) override;
		TaskCoroutine ExecuteAsync(TaskExecutionMode mode) override;
	private:
		PythonUDFChannel &channel;
		UDFBatch &batch;
};

} // namespace duckdb