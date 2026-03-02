Upload to EC2 using the command: scp -i DexcrowLogin.pem Projects/SlangShift/CCAnalysis/word_count.py ec2-user@100.55.27.135:~/

make sure to upload the target_word.txt file as well.

Download the result file using:
scp -i DexcrowLogin.pem ec2-user@100.55.27.135:~/wet_counts_2013_20.json ~/Documents/Projects/SlangShift/CCAnalysis
