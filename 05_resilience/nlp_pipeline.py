
PROJECT='ai-analytics-solutions'
BUCKET='ai-analytics-solutions-kfpdemo'
REGION='us-central1'

from datetime import datetime
import apache_beam as beam

def parse_nlp_result(response):
    return [
        # response, # entire string
        response.sentences[0].text.content,
        response.language,
        response.document_sentiment.score
    ]

def run():
    from apache_beam.ml.gcp import naturallanguageml as nlp
    
    features = nlp.types.AnnotateTextRequest.Features(
        extract_entities=True,
        extract_document_sentiment=True,
        extract_syntax=False
    )
    options = beam.options.pipeline_options.PipelineOptions()
    google_cloud_options = options.view_as(beam.options.pipeline_options.GoogleCloudOptions)
    google_cloud_options.project = PROJECT
    google_cloud_options.region = REGION
    google_cloud_options.job_name = 'nlpapi-{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    google_cloud_options.staging_location = 'gs://{}/staging'.format(BUCKET)
    google_cloud_options.temp_location = 'gs://{}/temp'.format(BUCKET)
    options.view_as(beam.options.pipeline_options.StandardOptions).runner = 'DataflowRunner' # 'DirectRunner'

    p = beam.Pipeline(options=options)
    (p 
     | 'bigquery' >> beam.io.Read(beam.io.BigQuerySource(
         query="SELECT text FROM `bigquery-public-data.hacker_news.comments` WHERE author = 'AF' AND LENGTH(text) > 10",
         use_standard_sql=True))
      | 'txt'      >> beam.Map(lambda x : x['text'])
      | 'doc'      >> beam.Map(lambda x : nlp.Document(x, type='PLAIN_TEXT'))
    #  | 'todict'   >> beam.Map(lambda x : nlp.Document.to_dict(x))
      | 'nlp'      >> nlp.AnnotateText(features, timeout=10)
      | 'parse'    >> beam.Map(parse_nlp_result)
      | 'gcs'      >> beam.io.WriteToText('gs://{}/output.txt'.format(BUCKET), num_shards=1)
    )
    result = p.run()
    result.wait_until_finish()

if __name__ == '__main__':
    run()
