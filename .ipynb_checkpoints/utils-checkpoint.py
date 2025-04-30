import boto3
import json
import datetime
import uuid

class BedrockBatchProcessor:
    def __init__(self, bucket, role_arn, model_id, output_path, key_prefix):
        """
        Initialize the BedrockBatchProcessor with required configuration.
        
        Args:
            bucket (str): S3 bucket name for storing chunks
            role_arn (str): IAM role ARN with necessary permissions
            model_id (str): Bedrock model ID to use for processing
            output_path (str): S3 path where output will be stored
            key_prefix (str): Prefix for S3 keys
        """
        self.bucket = bucket
        self.role_arn = role_arn
        self.model_id = model_id
        self.output_path = output_path
        self.key_prefix = key_prefix
        self.s3_client = boto3.client('s3')
        self.bedrock_client = boto3.client('bedrock', region_name = 'us-east-1')
    
    def chunk_list(self, data_list, chunk_size=50000):
        """
        Split a large list into smaller chunks.
        
        Args:
            data_list (list): The list to be chunked
            chunk_size (int): Maximum size of each chunk
            
        Returns:
            list: List of chunks
        """
        return [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]
    
    def upload_chunk_to_s3(self, data_chunk, chunk_index):
        """
        Convert a chunk to JSONL and upload to S3.
        
        Args:
            data_chunk (list): The chunk to upload
            chunk_index (int): Index of the chunk for naming
            
        Returns:
            str: S3 key of the uploaded chunk
        """
        # Convert list of JSON objects to JSONL format
        jsonl_data = '\n'.join([json.dumps(item) for item in data_chunk])
        
        # Create a unique key for this chunk
        key = f"{self.key_prefix}_chunk_{chunk_index}.jsonl"
        
        # Upload to S3
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=jsonl_data
        )
        
        return key
    
    def generate_job_name(self, prefix="batch"):
        """
        Generate a compliant job name for Bedrock.
        
        Args:
            prefix (str): Prefix for the job name
            
        Returns:
            str: A unique job name
        """
        # Get current date and time
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        
        # Generate a short unique ID
        short_uuid = str(uuid.uuid4()).replace('-', '')[:8]
        
        # Combine to create a unique job name
        job_name = f"{prefix}-{timestamp}-{short_uuid}"
        
        return job_name
    
    def get_output_data_config(self, job_name):
        """
        Create the output data configuration for Bedrock.
        
        Args:
            job_name (str): Name of the job
            
        Returns:
            dict: Output data configuration
        """
        return {"s3OutputDataConfig": {"s3Uri": f"{self.output_path}/{job_name}/"}}
    
    def create_batch_job(self, input_s3_uri, job_name=None):
        """
        Create a batch job in Bedrock.
        
        Args:
            input_s3_uri (str): S3 URI of the input data
            job_name (str, optional): Custom job name. If None, one will be generated
            
        Returns:
            tuple: (job_arn, job_name)
        """
        if job_name is None:
            job_name = self.generate_job_name()
        
        input_data_config = {
            "s3InputDataConfig": {
                "s3Uri": input_s3_uri
            }
        }
        
        response = self.bedrock_client.create_model_invocation_job(
            roleArn=self.role_arn,
            modelId=self.model_id,
            jobName=job_name,
            inputDataConfig=input_data_config,
            outputDataConfig=self.get_output_data_config(job_name)
        )
        
        return response.get('jobArn'), job_name
    
    def process_data(self, data, chunk_size=40000):
        """
        Process data by chunking, uploading to S3, and creating batch jobs.
        
        Args:
            data (list): List of data to process
            chunk_size (int): Size of each chunk
            
        Returns:
            list: List of tuples containing (job_arn, job_name) for each job
        """
        chunks = self.chunk_list(data, chunk_size)
        jobs = []
        
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i+1}/{len(chunks)}")
            
            # Upload chunk to S3
            s3_key = self.upload_chunk_to_s3(chunk, i)
            s3_uri = f"s3://{self.bucket}/{s3_key}"
            
            # Create batch job for this chunk
            job_arn, job_name = self.create_batch_job(s3_uri)
            jobs.append((job_arn, job_name))
            
            print(f"Created job: {job_name} with ARN: {job_arn}")
        
        return jobs