import torch
from torch.utils.data import DataLoader # 데이터로더
from gluonnlp.data import SentencepieceTokenizer 
# from kogpt2.utils import get_tokenizer
# from kogpt2.utils import download, tokenizer
from kogpt2.model.torch_gpt2 import GPT2LMHeadModel
from transformers.configuration_gpt2 import GPT2Config
from kogpt2.data import Read_Dataset
import gluonnlp
from kogpt2.model.sample import sample_sequence
from tqdm import tqdm
import subprocess
from tensorboardX import SummaryWriter
import re
import copy
import logging
# import dropbox.files
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
import os
import sys
import json

#드롭 박스를 이용하기 위해서 쓰는 코드 드롭박스 2T를 쓰기 위해서 선언
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
my_drive = GoogleDrive(gauth)

# 각자의 드롭박스 key 입력
# FORUS_AI_RESOURCES_APP_ACCESS_TOKEN =  ''

# logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
#                     datefmt='%Y-%m-%d:%H:%M:%S',
#                     level=logging.WARNING)
# logger = logging.getLogger(__name__)
# dbx = dropbox.Dropbox(FORUS_AI_RESOURCES_APP_ACCESS_TOKEN)

def auto_enter(text):
	text = text.replace("   ", "\n")
	text = text.split("\n")
	text = [t.lstrip() for t in text if t != '']
	text = "\n\n".join(text)
	text = text.replace("  ", "")
	return text

def get_gpu_memory_map():
	"""Get the current gpu usage.

	Returns
	-------
	usage: dict
		Keys are device ids as integers.
		Values are memory usage as integers in MB.
	"""
	result = subprocess.check_output(
		[
			'nvidia-smi', '--query-gpu=memory.used',
			'--format=csv,nounits,noheader'
		], encoding='utf-8')
	# Convert lines into a dictionary
	gpu_memory = [int(x) for x in result.strip().split('\n')]
	gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
	return gpu_memory_map

def main(epoch = 200, save_path = './checkpoint/', load_path = './checkpoint/KoGPT2_checkpoint_long.tar',
		 data_file_path = 'dataset/lyrics_dataset.txt',
		 batch_size = 8, summary_url = 'runs/', new = 0, text_size = 100):
	ctx = 'cuda'
	cachedir = '~/kogpt2/'
	summary = SummaryWriter(summary_url)

	# pytorch_kogpt2 = {
	# 	'url': 'https://kobert.blob.core.windows.net/models/kogpt2/pytorch/pytorch_kogpt2_676e9bcfa7.params',
	# 	'fname': 'pytorch_kogpt2_676e9bcfa7.params',
	# 	'chksum': '676e9bcfa7'
	# }
	kogpt2_config = {
		"initializer_range": 0.02,
		"layer_norm_epsilon": 1e-05,
		"n_ctx": 1024,
		"n_embd": 768,
		"n_head": 12,
		"n_layer": 12,
		"n_positions": 1024,
		"vocab_size": 50000
	}

	# download model
	# model_info = pytorch_kogpt2
	# model_path = download(model_info['url'],
	# 					   model_info['fname'],
	# 					   model_info['chksum'],
	# 					   cachedir=cachedir)
	# download vocab
	# vocab_info = tokenizer
	# vocab_path = download(vocab_info['url'],
	# 					   vocab_info['fname'],
	# 					   vocab_info['chksum'],
	# 					   cachedir=cachedir)

	model_path = 'config.json'

	# KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
	kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))

	# model_path 로부터 다운로드 받은 내용을 load_state_dict 으로 업로드
	# 기본 모델에서 가져오는 파라미터 업데이트
	kogpt2model.load_state_dict(json.load(open(model_path)), strict=False)

	device = torch.device(ctx) #GPU
	kogpt2model.to(device)
	count = 0

	# 체크포인트에서 불러오기 부분
	try:
		checkpoint = torch.load(load_path, map_location=device)

		# KoGPT-2 언어 모델 학습을 위한 GPT2LMHeadModel 선언
		kogpt2model = GPT2LMHeadModel(config=GPT2Config.from_dict(kogpt2_config))
		kogpt2model.load_state_dict(checkpoint['model_state_dict'])

		kogpt2model.eval()
	except:
		print("count 0 : ", load_path)
	else:
		print("count check : ",re.findall("\d+", load_path))
		count = max([int(i) for i in (re.findall("\d+", load_path))])

	if new:
		count = 0

	# 추가로 학습하기 위해 .train() 사용
	kogpt2model.train()
	vocab_b_obj = gluonnlp.vocab.BERTVocab.from_sentencepiece('kobert_news_wiki_ko_cased-1087f8699e.spiece',
														 mask_token=None,
														 sep_token=None,
														 cls_token=None,
														 unknown_token='[UNK]',
														 padding_token='[PAD]',
														 bos_token='<s>',
														 eos_token='</s>')

	# tok_path = get_tokenizer()
	model, vocab = kogpt2model, vocab_b_obj
	tok = SentencepieceTokenizer('kobert_news_wiki_ko_cased-1087f8699e.spiece')

	# 우리의 데이터셋 불러오는 부분
	dataset = Read_Dataset(data_file_path, vocab, tok)
	data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)

	# 체크
	learning_rate = 5e-5
	# criterion = torch.nn.CrossEntropyLoss()
	optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

	model = model.to(ctx)
	# bpe로 할 때 나누고 합치고 하는 과정이 편해짐.

	print('KoGPT-2 Transfer Learning Start')

	# 장르별로 체크포인트 폴더 없으면 생성하기
	try:
		if not(os.path.isdir(save_path + data_file_path.split("/")[-1][:-4])):
			os.makedirs(os.path.join(save_path + data_file_path.split("/")[-1][:-4]))
	except OSError as e:
		if e.errno != errno.EEXIST:
			print("Failed to create directory!!!!!")
			raise
	
	avg_loss = (0.0, 0.0)
	for epoch in range(epoch):
		# 데이터셋 가져와서 학습 시작
		for datas in data_loader:
			data = datas[0]

			optimizer.zero_grad()
			data = torch.stack(data) # list of Tensor로 구성되어 있기 때문에 list를 stack을 통해 변환해준다.
			data = data.transpose(1,0)
			# print(data)
			data = data.to(ctx)
			model = model.to(ctx)

			# 실제 학습
			outputs = model(data, labels=data)
			loss, logits = outputs[:2]

			nowloss = copy.copy(loss)
			# 평균 loss 만들기 avg_loss[0] / avg_loss[1] <- loss 정규화
			avg_loss = (avg_loss[0]`` * 0.99 + loss, avg_loss[1] * 0.99 + 1.0)

			loss *= datas[2][0] # 특별 socre 부분

			loss = loss.to(ctx)
			loss.backward()
			
			# 학습 끝
			optimizer.step()

			if count % 10 == 0:
				print('epoch no.{0} train no.{1}  loss = {2:.5f} avg_loss = {3:.5f}' . format(epoch, count, loss, avg_loss[0] / avg_loss[1]))
				summary.add_scalar('loss/avg_loss', avg_loss[0] / avg_loss[1], count)
				summary.add_scalar('loss/loss', loss, count)

				#generator 진행
				if (count > 0 and count % 500 == 0):
					sent = sample_sequence(model.to("cpu"), tok, vocab, sent="너라면", text_size=text_size, temperature=0.7, top_p=0.9, top_k=100)
					sent = sent.replace("//", "\n") # 비효율적이지만 엔터를 위해서 등장
					sent = auto_enter(sent)
					print(sent)
					summary.add_text('Text', sent, count)
					del sent
					pass

			#########################################
			if (count > 0 and count % 15000 == 0):
				print("모델을 저장합니다.")
				# 모델 저장
				try:
					torch.save(
                        model.state_dict(),
                    #     {
					# 	'epoch': epoch,
					# 	'train_no': count,
					# 	'model_state_dict': model.state_dict(),
					# 	'optimizer_state_dict': optimizer.state_dict(),
					# 	'loss': loss
					# }, 
                    save_path + data_file_path.split("/")[-1][:-4] + '/' + 'KoGPT2_checkpoint_' + str(count) + '.pth')

				except:
					pass

			if (avg_loss[0] / avg_loss[1] < 1.0):
				print("학습이 끝났어용!!")
				print("모델을 저장합니다.")
				# 모델 저장
				torch.save(
                    model.state_dict(),
                #     {
				# 	'epoch': epoch,
				# 	'train_no': count,
				# 	'model_state_dict': model.state_dict(),
				# 	'optimizer_state_dict': optimizer.state_dict(),
				# 	'loss': loss
				# }, 
                save_path + data_file_path.split("/")[-1][:-4] + '/' + 'KoGPT2_checkpoint_' + str(count) + '.pth')

			count += 1
			torch.cuda.empty_cache()
