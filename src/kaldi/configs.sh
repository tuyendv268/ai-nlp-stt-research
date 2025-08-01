# Database theo formet của Kaldi: Gồm 3 filr utt2spk, text, wav.scp
# Cần chú ý them số -r trong lệnh sox trong file wav.scp phải đúng với mục đích build hệ thống
# VD: cần build hệ thống cho dữ liệu 8k thì tham số có dạng -r 8000
# nêu là 16k thì: -r 16000
traindata=FPTFulldata-NoFRT-20190228
# Từ điển 
traning_lex=resources/train.dict
# Text corpus để build Language model
huge_text=Resources/Text_corpus/All.gz
noise_words="<HumanNoise> <Noise> <Music>"
noise_phonemes="+humannoise+ +noise+ +music+"
# Nếu hệ thống là 8k
data_kind="16k"
#data_kind="8k"
# Kết quả sau huấn luyện đặt ở đây
output_model="model_out"
