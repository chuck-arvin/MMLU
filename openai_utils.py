import pandas as pd

class OpenAIBatchRunner():
    def __init__(self, data, chunk_size = 50000):
        """Initialize the BatchProcessor with OpenAI client."""
        self.client = OpenAI()
        self.data = data
        self.chunk_size = chunk_size
        self.batch_job = None
        self.input_file_id = None
    
    def chunk_list(self, data, chunk_size):
        """
        Split a large list into smaller chunks.
        
        Args:
            data_list (list): The list to be chunked
            chunk_size (int): Maximum size of each chunk
            
        Returns:
            list: List of chunks
        """
        return [self.data[i:i + self.chunk_size] for i in range(0, len(self.data), self.chunk_size)]
    
    def upload_file(self, chunk):
        """Upload a JSONL file to OpenAI and return the file ID."""

        jsonl_data = '\n'.join([json.dumps(item) for item in chunk])

        with open("temp_batch_input.jsonl", "w") as f:
            for item in chunk:
                f.write(json.dumps(item) + "\n")
                
        # Upload the file
        batch_input_file = self.client.files.create(
            file=open("temp_batch_input.jsonl", "rb"),
            purpose="batch")
        
        return batch_input_file.id

    def create_batch_job(self, filename):
        """Create a batch job with the uploaded file."""
        batch_job = self.client.batches.create(
            input_file_id=filename,
            endpoint="/v1/chat/completions",
            completion_window="24h"
        )
        return batch_job

    def process_data(self, max_retries = 5):
        chunks = self.chunk_list(self.data, self.chunk_size)
        jobs = []

        for i, chunk in enumerate(chunks):
            for retry_attempt in range(max_retries):
                try:
                    filename = self.upload_file(chunk)
                    job = self.create_batch_job(filename)
                    jobs.append(job)
                    print(f"Created job: {job}")
                    break  # Exit the retry loop on success
                except Exception as e:
                    # Calculate backoff time (increasing with each retry)
                    wait_time = 2 ** retry_attempt  # Exponential backoff: 1, 2, 4, 8, 16 seconds
                    if retry_attempt < max_retries - 1:
                        print(f"Attempt {retry_attempt + 1} failed for chunk {i+1}. "
                              f"Retrying in {wait_time} seconds. Error: {str(e)}")
                        time.sleep(wait_time)
                    else:
                        print(f"Failed to process chunk {i+1} after {max_retries} attempts. "
                              f"Error: {str(e)}")
                        raise  # Re-raise the exception if we've exhausted all retries
                
        return jobs

def process_batchfile_openai(client, batch):
    batch = client.batches.retrieve(batch)
    file = batch.output_file_id
    dat = pd.read_json(client.files.content(file), lines = True)
    dat['model_id'] = dat['response'].apply(lambda x: x.get('body').get('model'))
    dat['model_response'] = dat['response'].apply(lambda x: x.get('body').get('choices')[0].get('message').get('content'))
    dat['input_tokens'] = dat['response'].apply(lambda x: x.get('body').get('usage').get('prompt_tokens'))
    dat['output_tokens'] = dat['response'].apply(lambda x: x.get('body').get('usage').get('completion_tokens'))
    return dat
