import time
import traceback
from functools import partial
from ollama_helper import query_model_ollama
try:
    from transformers_helper import TransformersInferencer, get_model_batch_size_from_config
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


INFERENCE_FRAMEWORKS = ["ollama", "transformers"]

INFERENCER = None

def do_transformers_inference(query_df, show_progress=True):
    nonempty_queries = query_df["query"].apply(bool)
    if nonempty_queries.any():
        batch_size = get_model_batch_size_from_config(query_df.loc[nonempty_queries]["query"].iloc[0]["model"])
        first_try = True
        while True:
            try:
                query_df.loc[nonempty_queries, "model_answer"] = INFERENCER.batch_query_model(
                    query_df.loc[nonempty_queries, "query"].tolist(), batch_size=batch_size,
                    show_progress=show_progress)
                return query_df
            except Exception as ex:
                INFERENCER.unload_model()
                if not first_try:
                    if batch_size == 1:
                        # error not fixable by reducing the batch size
                        traceback.print_exc()
                        # rethrow error
                        raise RuntimeError("Not possible to do inference, even with batch size 1") from ex
                    else:
                        # reduce batch size
                        batch_size = batch_size - 1
                        print(f"Error (probably memory error). Reducing batch size. New batchsize: {batch_size}")
                first_try = False
    else:
        return query_df


# executes all queries in the column "query" in the queries_df and saves the result in the column "model_answer" and the
# time for executing the query in the column "query_execution_time"
def execute_queries(query_df, inference_framework, batch_process=None, show_progress=True):

    global INFERENCER

    if inference_framework == "ollama":
        query_model = query_model_ollama
        if batch_process:
            raise ValueError("Batch processing not supported by ollama backend")
    elif inference_framework == "transformers":
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Dependencies for transformers package not available and need to be installed first")
        if not INFERENCER:
            INFERENCER = TransformersInferencer()
        if batch_process != False: # use batch processing by default in transformers framework
            return do_transformers_inference(query_df, show_progress=show_progress)
        else:
            query_model = partial(INFERENCER.query_model) # Continue with execution of single queries
    else:
        raise ValueError(f"Unknown inference framework {inference_framework}")

    total_nrows = len(query_df)
    i = 0
    model_answers = []
    query_execution_times = []

    for queryrow in query_df.itertuples():
        query_start_time = time.time()
        if queryrow.query:
            try:
                model_answers.append(query_model(queryrow.query))
            except Exception as e:
                model_answers.append("<error>")
                print(f'Error processing query {i+1}/{total_nrows}: {e}')

            query_end_time = time.time()
            query_execution_times.append(query_end_time - query_start_time)

            if show_progress:
                print(f'Query {i+1}/{total_nrows} finished in {query_execution_times[i]} seconds.', end='\r', flush=True)

        else: # empty query, do not execute
            model_answers.append(queryrow.model_answer) # answer must be present already
            query_execution_times.append(0)

            if show_progress:
                print(f'Query {i+1}/{total_nrows} is empty: skipped.', end='\r', flush=True)

        if show_progress:
            if i + 1 == total_nrows:
                print('\n')

        i += 1 # increment progress
    query_df["model_answer"] = model_answers
    query_df["query_execution_time"] = query_execution_times
    return query_df