#include "duckdb/main/query_result.hpp"
#include "duckdb_python/pybind11/pybind_wrapper.hpp"
#include "duckdb/function/scalar_function.hpp"
#include "duckdb_python/pytype.hpp"
#include "duckdb_python/pyconnection/pyconnection.hpp"
#include "duckdb_python/pandas/pandas_scan.hpp"
#include "duckdb/common/arrow/arrow.hpp"
#include "duckdb/common/arrow/arrow_converter.hpp"
#include "duckdb/common/arrow/arrow_wrapper.hpp"
#include "duckdb/common/arrow/arrow_appender.hpp"
#include "duckdb/common/arrow/result_arrow_wrapper.hpp"
#include "duckdb_python/arrow/arrow_array_stream.hpp"
#include "duckdb/function/table/arrow.hpp"
#include "duckdb/function/function.hpp"
#include "duckdb_python/numpy/numpy_scan.hpp"
#include "duckdb_python/arrow/arrow_export_utils.hpp"
#include "duckdb/common/types/arrow_aux_data.hpp"
#include "duckdb/parser/tableref/table_function_ref.hpp"
#include "duckdb/function/table/arrow/arrow_duck_schema.hpp"
#include "duckdb_python/python_conversion.hpp"

#include "duckdb_python/python_udf_channel.hpp"
#include "duckdb/parallel/task_scheduler.hpp"

namespace duckdb {

static py::list ConvertToSingleBatch(vector<LogicalType> &types, vector<string> &names, DataChunk &input,
                                     ClientProperties &options, ClientContext &context) {
	ArrowSchema schema;
	ArrowConverter::ToArrowSchema(&schema, types, names, options);

	py::list single_batch;
	ArrowAppender appender(types, STANDARD_VECTOR_SIZE, options,
	                       ArrowTypeExtensionData::GetExtensionTypes(context, types));
	appender.Append(input, 0, input.size(), input.size());
	auto array = appender.Finalize();
	TransformDuckToArrowChunk(schema, array, single_batch);
	return single_batch;
}

static py::object ConvertDataChunkToPyArrowTable(DataChunk &input, ClientProperties &options, ClientContext &context) {
	auto types = input.GetTypes();
	vector<string> names;
	names.reserve(types.size());
	for (idx_t i = 0; i < types.size(); i++) {
		names.push_back(StringUtil::Format("c%d", i));
	}

	return pyarrow::ToArrowTable(types, names, ConvertToSingleBatch(types, names, input, options, context), options);
}

// If these types are arrow canonical extensions, we must check if they are registered.
// If not, we should error.
void AreExtensionsRegistered(const LogicalType &arrow_type, const LogicalType &duckdb_type) {
	if (arrow_type != duckdb_type) {
		// Is it a UUID Registration?
		if (arrow_type.id() == LogicalTypeId::BLOB && duckdb_type.id() == LogicalTypeId::UUID) {
			throw InvalidConfigurationException(
			    "Mismatch on return type from Arrow object (%s) and DuckDB (%s). It seems that you are using the UUID "
			    "arrow canonical extension, but the same is not yet registered. Make sure to register it first with "
			    "e.g., pa.register_extension_type(UUIDType()). ",
			    arrow_type.ToString(), duckdb_type.ToString());
		}
		// Is it a JSON Registration
		if (!arrow_type.IsJSONType() && duckdb_type.IsJSONType()) {
			throw InvalidConfigurationException(
			    "Mismatch on return type from Arrow object (%s) and DuckDB (%s). It seems that you are using the JSON "
			    "arrow canonical extension, but the same is not yet registered. Make sure to register it first with "
			    "e.g., pa.register_extension_type(JSONType()). ",
			    arrow_type.ToString(), duckdb_type.ToString());
		}
	}
}
static void ConvertArrowTableToVector(const py::object &table, Vector &out, ClientContext &context, idx_t count) {
	// Create the stream factory from the Table object
	auto ptr = table.ptr();
	D_ASSERT(py::gil_check());
	py::gil_scoped_release gil;

	auto stream_factory =
	    make_uniq<PythonTableArrowArrayStreamFactory>(ptr, context.GetClientProperties(), PyArrowObjectType::Table);
	auto stream_factory_produce = PythonTableArrowArrayStreamFactory::Produce;
	auto stream_factory_get_schema = PythonTableArrowArrayStreamFactory::GetSchema;

	// Get the functions we need
	auto function = ArrowTableFunction::ArrowScanFunction;
	auto bind = ArrowTableFunction::ArrowScanBind;
	auto init_global = ArrowTableFunction::ArrowScanInitGlobal;
	auto init_local = ArrowTableFunction::ArrowScanInitLocalInternal;

	// Prepare the inputs for the bind
	vector<Value> children;
	children.reserve(3);
	children.push_back(Value::POINTER(CastPointerToValue(stream_factory.get())));
	children.push_back(Value::POINTER(CastPointerToValue(stream_factory_produce)));
	children.push_back(Value::POINTER(CastPointerToValue(stream_factory_get_schema)));
	named_parameter_map_t named_params;
	vector<LogicalType> input_types;
	vector<string> input_names;

	TableFunctionRef empty;
	TableFunction dummy_table_function;
	dummy_table_function.name = "ConvertArrowTableToVector";
	TableFunctionBindInput bind_input(children, named_params, input_types, input_names, nullptr, nullptr,
	                                  dummy_table_function, empty);
	vector<LogicalType> return_types;
	vector<string> return_names;

	auto bind_data = bind(context, bind_input, return_types, return_names);

	if (return_types.size() != 1) {
		throw InvalidInputException(
		    "The returned table from a pyarrow scalar udf should only contain one column, found %d",
		    return_types.size());
	}

	AreExtensionsRegistered(return_types[0], out.GetType());

	DataChunk result;
	// Reserve for STANDARD_VECTOR_SIZE instead of count, in case the returned table contains too many tuples
	result.Initialize(context, return_types, STANDARD_VECTOR_SIZE);

	vector<column_t> column_ids = {0};
	TableFunctionInitInput input(bind_data.get(), column_ids, vector<idx_t>(), nullptr);
	auto global_state = init_global(context, input);
	auto local_state = init_local(context, input, global_state.get());

	TableFunctionInput function_input(bind_data.get(), local_state.get(), global_state.get());
	function(context, function_input, result);
	if (result.size() != count) {
		throw InvalidInputException("Returned pyarrow table should have %d tuples, found %d", count, result.size());
	}

	VectorOperations::Cast(context, result.data[0], out, count);
	out.Flatten(count);
	out.Verify(count);
}

static string NullHandlingError() {
	return R"(
The returned result contained NULL values, but the 'null_handling' was set to DEFAULT.
If you want more control over NULL values then 'null_handling' should be set to SPECIAL.

With DEFAULT all rows containing NULL have been filtered from the UDFs input.
Those rows are automatically set to NULL in the final result.
The UDF is not expected to return NULL values.
	)";
}

static ValidityMask &GetResultValidity(Vector &result) {
	auto vector_type = result.GetVectorType();
	if (vector_type == VectorType::CONSTANT_VECTOR) {
		return ConstantVector::Validity(result);
	} else if (vector_type == VectorType::FLAT_VECTOR) {
		return FlatVector::Validity(result);
	} else {
		throw InternalException("VectorType %s was not expected here (GetResultValidity)",
		                        EnumUtil::ToString(vector_type));
	}
}

static void VerifyVectorizedNullHandling(Vector &result, idx_t count) {
	auto &validity = GetResultValidity(result);

	if (validity.AllValid()) {
		return;
	}

	throw InvalidInputException(NullHandlingError());
}

static scalar_function_t CreateVectorizedFunction(PyObject *function, PythonExceptionHandling exception_handling,
                                                  FunctionNullHandling null_handling,
												  shared_ptr<PythonUDFChannel> channel) {
	// Through the capture of the lambda, we have access to the function pointer
	// We just need to make sure that it doesn't get garbage collected
	scalar_function_t func = [=](DataChunk &input, ExpressionState &state, Vector &result) -> void {
		auto &context = state.GetContext();
		auto options = context.GetClientProperties();
 
		const bool default_null_handling = null_handling == FunctionNullHandling::DEFAULT_NULL_HANDLING;
 
		// ---- NULL HANDLING (same as original) ----
		auto result_validity = FlatVector::Validity(result);
		SelectionVector selvec(input.size());
		idx_t input_size = input.size();
		idx_t count = input_size;
 
		if (default_null_handling) {
			vector<UnifiedVectorFormat> vec_data(input.ColumnCount());
			for (idx_t i = 0; i < input.ColumnCount(); i++) {
				input.data[i].ToUnifiedFormat(input.size(), vec_data[i]);
			}
 
			idx_t index = 0;
			for (idx_t i = 0; i < input.size(); i++) {
				bool any_null = false;
				for (idx_t col_idx = 0; col_idx < input.ColumnCount(); col_idx++) {
					auto &vec = vec_data[col_idx];
					if (!vec.validity.RowIsValid(vec.sel->get_index(i))) {
						any_null = true;
						break;
					}
				}
				if (any_null) {
					result_validity.SetInvalid(i);
					continue;
				}
				selvec.set_index(index++, i);
			}
			if (index != input.size()) {
				input.Slice(selvec, index);
			}
			count = input.size();
		}
 
		// ---- CONVERT DataChunk TO Arrow C Data Interface ----
		auto types = input.GetTypes();
		vector<string> names;
		names.reserve(types.size());
		for (idx_t i = 0; i < types.size(); i++) {
			names.push_back(StringUtil::Format("c%d", i));
		}
 
		ArrowSchema *schema = new ArrowSchema();
		ArrowConverter::ToArrowSchema(schema, types, names, options);
 
		ArrowAppender appender(types, STANDARD_VECTOR_SIZE, options,
		                       ArrowTypeExtensionData::GetExtensionTypes(context, types));
		appender.Append(input, 0, count, count);
		auto array_result = appender.Finalize();
 
		// Copy the ArrowArray to a heap-allocated struct that the channel owns.
		ArrowArray *arr = new ArrowArray();
		memcpy(arr, &array_result, sizeof(ArrowArray));
		// Prevent the original from calling release — ownership transferred.
		array_result.release = nullptr;
 
		// ---- PUSH THROUGH CHANNEL ----
		ArrowBatch batch{arr, schema};
		auto task = make_shared_ptr<PythonUDFTask>(*channel, std::move(batch));
 
		auto &scheduler = TaskScheduler::GetScheduler(context);
		auto coroutine = task->ExecuteAsync(TaskExecutionMode::PROCESS_ALL);
		coroutine.set_scheduler(&scheduler);
		coroutine.resume();
 
		// Block until the coroutine completes.
		// Yield the thread rather than spinning hot.
		while (!coroutine.done()) {
			TaskScheduler::YieldThread();
		}
 
		if (channel->HasError()) {
			throw InvalidInputException("Python UDF error: %s", channel->GetError());
		}
 
		auto task_result = coroutine.result();
		if (task_result == TaskExecutionResult::TASK_ERROR) {
			throw InvalidInputException("Python UDF task failed");
		}
 
		// ---- CONVERT RESULT Arrow BACK TO Vector ----
		auto &result_batch = task->GetResult();

		// Build a one-shot ArrowArrayStream from the raw C Data Interface structs.
		// This feeds into DuckDB's existing Arrow scan without touching Python.
		struct SingleBatchStreamState {
			ArrowArray *batch;
			ArrowSchema *schema;
			bool schema_consumed = false;
			bool batch_consumed = false;
		};

		auto stream_state = new SingleBatchStreamState{result_batch.array, result_batch.schema};

		ArrowArrayStream arrow_stream;
		arrow_stream.private_data = stream_state;

		arrow_stream.get_schema = [](ArrowArrayStream *stream, ArrowSchema *out) -> int {
			auto *st = static_cast<SingleBatchStreamState *>(stream->private_data);
			if (!st->schema) return -1;
			*out = *st->schema;
			st->schema_consumed = true;
			return 0;
		};

		arrow_stream.get_next = [](ArrowArrayStream *stream, ArrowArray *out) -> int {
			auto *st = static_cast<SingleBatchStreamState *>(stream->private_data);
			if (st->batch_consumed || !st->batch) {
				out->release = nullptr;
				return 0;
			}
			*out = *st->batch;
			st->batch_consumed = true;
			return 0;
		};

		arrow_stream.get_last_error = [](ArrowArrayStream *stream) -> const char * {
			return nullptr;
		};

		arrow_stream.release = [](ArrowArrayStream *stream) {
			auto *st = static_cast<SingleBatchStreamState *>(stream->private_data);
			delete st;
			stream->private_data = nullptr;
			stream->release = nullptr;
		};

		// Use the existing Arrow scan bind/init/scan to convert to DataChunk.
		auto stream_factory_produce = [](uintptr_t factory_ptr, ArrowStreamParameters &params) -> unique_ptr<ArrowArrayStreamWrapper> {
			auto wrapper = make_uniq<ArrowArrayStreamWrapper>();
			auto *stream_ptr = reinterpret_cast<ArrowArrayStream *>(factory_ptr);
			wrapper->arrow_array_stream = *stream_ptr;
			stream_ptr->release = nullptr;
			return wrapper;
		};

		auto stream_factory_get_schema = [](uintptr_t factory_ptr, ArrowSchemaWrapper &schema) {
			auto *stream_ptr = reinterpret_cast<ArrowArrayStream *>(factory_ptr);
			stream_ptr->get_schema(stream_ptr, &schema.arrow_schema);
		};

		vector<Value> scan_children;
		scan_children.push_back(Value::POINTER(CastPointerToValue(&arrow_stream)));
		scan_children.push_back(Value::POINTER(CastPointerToValue(+stream_factory_produce)));
		scan_children.push_back(Value::POINTER(CastPointerToValue(+stream_factory_get_schema)));

		named_parameter_map_t scan_named_params;
		vector<LogicalType> scan_input_types;
		vector<string> scan_input_names;

		TableFunctionRef scan_empty;
		TableFunction scan_dummy;
		scan_dummy.name = "ArrowResultScan";
		TableFunctionBindInput scan_bind_input(scan_children, scan_named_params, scan_input_types,
		                                       scan_input_names, nullptr, nullptr, scan_dummy, scan_empty);
		vector<LogicalType> return_types;
		vector<string> return_names;

		auto scan_bind_data = ArrowTableFunction::ArrowScanBind(context, scan_bind_input, return_types, return_names);

		if (return_types.size() != 1) {
			arrow_stream.release(&arrow_stream);
			throw InvalidInputException(
			    "The returned table from a pyarrow scalar udf should only contain one column, found %d",
			    return_types.size());
		}

		DataChunk result_chunk;
		result_chunk.Initialize(context, return_types, STANDARD_VECTOR_SIZE);

		vector<column_t> scan_column_ids = {0};
		TableFunctionInitInput scan_init_input(scan_bind_data.get(), scan_column_ids, vector<idx_t>(), nullptr);
		auto scan_global_state = ArrowTableFunction::ArrowScanInitGlobal(context, scan_init_input);
		auto scan_local_state = ArrowTableFunction::ArrowScanInitLocalInternal(context, scan_init_input,
		                                                                        scan_global_state.get());

		TableFunctionInput scan_function_input(scan_bind_data.get(), scan_local_state.get(),
		                                       scan_global_state.get());
		ArrowTableFunction::ArrowScanFunction(context, scan_function_input, result_chunk);

		if (result_chunk.size() != count) {
			throw InvalidInputException("Returned pyarrow table should have %d tuples, found %d",
			                            count, result_chunk.size());
		}

		// Arrow memory is owned by the stream and freed when the scan completes.
		// The stream's release callback deletes the SingleBatchStreamState.
		// We still need to free the outer structs we allocated.
		delete result_batch.array;
		delete result_batch.schema;
		result_batch.array = nullptr;
		result_batch.schema = nullptr;
 
		// ---- REINSERT NULLS (same logic as original) ----
		if (count != input_size) {
			D_ASSERT(default_null_handling);
			Vector temp(result.GetType(), count);
			VectorOperations::Cast(context, result_chunk.data[0], temp, count);
			if (count) {
				SelectionVector inverted(input_size);
				idx_t src_index = 0;
				for (idx_t i = 0; i < input_size; i++) {
					inverted.set_index(i, src_index);
					if (src_index + 1 < count && selvec.get_index(src_index) == i) {
						src_index++;
					}
				}
				VectorOperations::Copy(temp, result, inverted, count, 0, 0, input_size);
			}
			for (idx_t i = 0; i < input_size; i++) {
				FlatVector::SetNull(result, i, !result_validity.RowIsValid(i));
			}
			result.Verify(input_size);
		} else {
			VectorOperations::Cast(context, result_chunk.data[0], result, count);
		}
 
		if (input_size == 1) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
	};
	return func;
}

static scalar_function_t CreateNativeFunction(PyObject *function, PythonExceptionHandling exception_handling,
                                              const ClientProperties &client_properties,
                                              FunctionNullHandling null_handling) {
	// Through the capture of the lambda, we have access to the function pointer
	// We just need to make sure that it doesn't get garbage collected
	scalar_function_t func = [=](DataChunk &input, ExpressionState &state, Vector &result) -> void { // NOLINT
		py::gil_scoped_acquire gil;

		const bool default_null_handling = null_handling == FunctionNullHandling::DEFAULT_NULL_HANDLING;

		for (idx_t row = 0; row < input.size(); row++) {

			py::object ret;
			if (input.ColumnCount() > 0) {
				auto bundled_parameters = py::tuple((int)input.ColumnCount());
				bool contains_null = false;
				for (idx_t i = 0; i < input.ColumnCount(); i++) {
					// Fill the tuple with the arguments for this row
					auto &column = input.data[i];
					auto value = column.GetValue(row);
					if (value.IsNull() && default_null_handling) {
						contains_null = true;
						break;
					}
					bundled_parameters[i] = PythonObject::FromValue(value, column.GetType(), client_properties);
				}
				if (contains_null) {
					// Immediately insert None, no need to call the function
					FlatVector::SetNull(result, row, true);
					continue;
				}
				// Call the function
				ret = py::reinterpret_steal<py::object>(PyObject_CallObject(function, bundled_parameters.ptr()));
			} else {
				ret = py::reinterpret_steal<py::object>(PyObject_CallObject(function, nullptr));
			}

			if (!ret || ret.is_none()) {
				if (PyErr_Occurred()) {
					if (exception_handling == PythonExceptionHandling::FORWARD_ERROR) {
						auto exception = py::error_already_set();
						throw InvalidInputException("Python exception occurred while executing the UDF: %s",
						                            exception.what());
					}
					if (exception_handling == PythonExceptionHandling::RETURN_NULL) {
						PyErr_Clear();
						FlatVector::SetNull(result, row, true);
						continue;
					}
					throw NotImplementedException("Exception handling type not implemented");
				}
				if (default_null_handling) {
					throw InvalidInputException(NullHandlingError());
				}
			}
			TransformPythonObject(ret, result, row);
		}

		if (input.size() == 1) {
			result.SetVectorType(VectorType::CONSTANT_VECTOR);
		}
	};
	return func;
}

namespace {

struct ParameterKind {
	enum class Type : uint8_t { POSITIONAL_ONLY, POSITIONAL_OR_KEYWORD, VAR_POSITIONAL, KEYWORD_ONLY, VAR_KEYWORD };
	static ParameterKind::Type FromString(const string &type_str) {
		if (type_str == "POSITIONAL_ONLY") {
			return Type::POSITIONAL_ONLY;
		} else if (type_str == "POSITIONAL_OR_KEYWORD") {
			return Type::POSITIONAL_OR_KEYWORD;
		} else if (type_str == "VAR_POSITIONAL") {
			return Type::VAR_POSITIONAL;
		} else if (type_str == "KEYWORD_ONLY") {
			return Type::KEYWORD_ONLY;
		} else if (type_str == "VAR_KEYWORD") {
			return Type::VAR_KEYWORD;
		} else {
			throw NotImplementedException("ParameterKindType not implemented for '%s'", type_str);
		}
	}
};

static bool NumpyDeprecatesAccessToCore(const py::tuple &numpy_version) {
	if (numpy_version.empty()) {
		return false;
	}
	if (string(py::str(numpy_version[0])) == string("2")) {
		//! Starting with numpy version 2.0.0 the use of 'core' is deprecated.
		return true;
	}
	return false;
}

struct PythonUDFData {
public:
	PythonUDFData(const string &name, bool vectorized, FunctionNullHandling null_handling)
	    : name(name), null_handling(null_handling), vectorized(vectorized) {
		return_type = LogicalType::INVALID;
		param_count = DConstants::INVALID_INDEX;
	}

public:
	string name;
	vector<LogicalType> parameters;
	LogicalType return_type;
	LogicalType varargs = LogicalTypeId::INVALID;
	FunctionNullHandling null_handling;
	idx_t param_count;
	bool vectorized;

public:
	void Verify() {
		if (return_type == LogicalType::INVALID) {
			throw InvalidInputException("Could not infer the return type, please set it explicitly");
		}
	}

	void OverrideReturnType(const shared_ptr<DuckDBPyType> &type) {
		if (!type) {
			return;
		}
		return_type = type->Type();
	}

	void OverrideParameters(const py::object &parameters_p) {
		if (py::none().is(parameters_p)) {
			return;
		}
		if (!py::isinstance<py::list>(parameters_p)) {
			throw InvalidInputException("Either leave 'parameters' empty, or provide a list of DuckDBPyType objects");
		}

		auto params = py::list(parameters_p);
		if (params.size() != param_count) {
			throw InvalidInputException("%d types provided, but the provided function takes %d parameters",
			                            params.size(), param_count);
		}
		D_ASSERT(parameters.empty() || parameters.size() == param_count);
		if (parameters.empty()) {
			for (idx_t i = 0; i < param_count; i++) {
				parameters.push_back(LogicalType::ANY);
			}
		}
		idx_t i = 0;
		for (auto &param : params) {
			auto type = py::cast<shared_ptr<DuckDBPyType>>(param);
			parameters[i++] = type->Type();
		}
	}

	py::object GetSignature(const py::object &udf) {
		const int32_t PYTHON_3_10_HEX = 0x030a00f0;
		auto python_version = PY_VERSION_HEX;

		auto signature_func = py::module_::import("inspect").attr("signature");
		if (python_version >= PYTHON_3_10_HEX) {
			return signature_func(udf, py::arg("eval_str") = true);
		} else {
			return signature_func(udf);
		}
	}

	void AnalyzeSignature(const py::object &udf) {
		auto signature = GetSignature(udf);
		auto sig_params = signature.attr("parameters");
		auto return_annotation = signature.attr("return_annotation");
		auto empty = py::module_::import("inspect").attr("Signature").attr("empty");
		if (!py::none().is(return_annotation) && !empty.is(return_annotation)) {
			shared_ptr<DuckDBPyType> pytype;
			if (py::try_cast<shared_ptr<DuckDBPyType>>(return_annotation, pytype)) {
				return_type = pytype->Type();
			}
		}
		param_count = py::len(sig_params);
		parameters.reserve(param_count);
		auto params = py::dict(sig_params);
		for (auto &item : params) {
			auto &value = item.second;
			shared_ptr<DuckDBPyType> pytype;
			if (py::try_cast<shared_ptr<DuckDBPyType>>(value.attr("annotation"), pytype)) {
				parameters.push_back(pytype->Type());
			} else {
				std::string kind = py::str(value.attr("kind"));
				auto parameter_kind = ParameterKind::FromString(kind);
				if (parameter_kind == ParameterKind::Type::VAR_POSITIONAL) {
					varargs = LogicalType::ANY;
				}
				parameters.push_back(LogicalType::ANY);
			}
		}
	}

	ScalarFunction GetFunction(const py::function &udf, PythonExceptionHandling exception_handling, bool side_effects,
	                           const ClientProperties &client_properties, ClientContext &context) {

		// Import this module, because importing this from a non-main thread causes a segfault

		auto &import_cache = *DuckDBPyConnection::ImportCache();
		py::handle core;
		auto numpy = import_cache.numpy();
		if (!numpy) {
			throw InvalidInputException("'numpy' is required for this operation, but it wasn't installed");
		}
		auto numpy_version = py::cast<py::tuple>(numpy.attr("__version__"));
		if (NumpyDeprecatesAccessToCore(numpy_version)) {
			core = numpy.attr("_core");
		} else {
			core = numpy.attr("core");
		}
		(void)core.attr("multiarray");

		scalar_function_t func;
		if (vectorized) {
			auto &scheduler = TaskScheduler::GetScheduler(context);
			auto udf_channel = make_shared_ptr<PythonUDFChannel>(udf.ptr(), scheduler);
			udf_channel->Start();
			func = CreateVectorizedFunction(udf.ptr(), exception_handling, null_handling, udf_channel);
		} else {
			func = CreateNativeFunction(udf.ptr(), exception_handling, client_properties, null_handling);
		}
		FunctionStability function_side_effects =
		    side_effects ? FunctionStability::VOLATILE : FunctionStability::CONSISTENT;
		ScalarFunction scalar_function(name, std::move(parameters), return_type, func, nullptr, nullptr, nullptr,
		                               nullptr, varargs, function_side_effects, null_handling);
		return scalar_function;
	}
};

} // namespace

ScalarFunction DuckDBPyConnection::CreateScalarUDF(const string &name, const py::function &udf,
                                                   const py::object &parameters,
                                                   const shared_ptr<DuckDBPyType> &return_type, bool vectorized,
                                                   FunctionNullHandling null_handling,
                                                   PythonExceptionHandling exception_handling, bool side_effects) {
	PythonUDFData data(name, vectorized, null_handling);
	auto &connection = con.GetConnection();

	data.AnalyzeSignature(udf);
	data.OverrideParameters(parameters);
	data.OverrideReturnType(return_type);
	data.Verify();
	return data.GetFunction(udf, exception_handling, side_effects, connection.context->GetClientProperties(), *connection.context);
}

} // namespace duckdb
