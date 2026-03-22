Upload to EC2 using the command: scp -i DexcrowLogin.pem Projects/SlangShift/CCAnalysis/word_count.py ec2-user@<IPV4>:~/
Note: IPV4 can change after restart so make sure to update.

Make sure to upload the target_words.txt file as well.

## Running the script

Input files are specified either as positional arguments (local paths, globs, or `s3://` URIs) or via `--file-list` with a gzip-compressed Common Crawl `.paths.gz` index.

```bash
# Count all words in a local WET file, output JSON
python word_count.py /data/file.wet.gz --output counts.json

# Count only target words from a file, using a glob pattern
python word_count.py /data/*.wet.gz --words target_words.txt --output counts.json

# Stream a single file from S3, output CSV
python word_count.py s3://commoncrawl/crawl-data/CC-MAIN-2024-10/segments/.../file.wet.gz \
    --output counts.csv --output-format csv

# Use a Common Crawl paths index file to process many S3 files, top 50k words only
python word_count.py --file-list s3://commoncrawl/crawl-data/CC-MAIN-2024-10/wet.paths.gz \
    --words target_words.txt --output counts.json --top 50000

# WARC files with HTML extraction, minimum word length 4, TSV output
python word_count.py /data/*.warc.gz --type warc --min-length 4 --output counts.tsv
```

Download the result file using:
scp -i DexcrowLogin.pem ec2-user@<IPV4>:~/counts.json ~/Documents/Projects/SlangShift/CCAnalysis
