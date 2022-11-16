# %%
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cgi
import json
import io
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
import threading
from urllib.parse import urlparse, parse_qs
import sys
import shutil
import time
from functools import wraps
from typing import Callable
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Input
from keras.callbacks import Callback
import re
import numpy as np
import base64

def timeit(func: Callable):
  @wraps(func)
  def wrapper(*args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return (end-start), result
  return wrapper

class CustomCallback(Callback):
  def __init__(self, callback):
    self.on_epoch_end_callback = callback

  def on_epoch_end(self, epoch, logs=None):
    if self.on_epoch_end_callback:
      self.on_epoch_end_callback(epoch, logs)

def prepare_text(text: str):
  text = text.replace('\ufeff', '').lower()  # убираем первый невидимый символ
  # заменяем все символы кроме кириллицы на пустые символы
  return re.sub(r'[^А-я ]', '', text)

text = None
with open('train_data_true', 'r', encoding='utf-8') as f:
  text = prepare_text(f.read())

training_lock = threading.Lock()

def train_model(input_text, batch_size=32, epochs=100, learning_rate=0.001, epoch_end_callback=None, training_finished_callback=None, verbose="auto"):
  if not training_lock.acquire(blocking=False):
    return False
  def inner():
    try:
      # преобразуем исходный текст в массив OHE
      data = tokenizer.texts_to_matrix(prepare_text(input_text))
      # так как мы предсказываем по трем символам - четвертый
      n = data.shape[0] - inp_chars
      X = np.array([data[i:i + inp_chars, :] for i in range(n)])
      Y = data[inp_chars:]  # предсказание следующего символа
      print(data.shape)
      model.compile(loss='categorical_crossentropy',
                    metrics=['accuracy'], optimizer=Adam(learning_rate))
      history = model.fit(X, Y, batch_size=batch_size, epochs=epochs,
                          shuffle=False, callbacks=[CustomCallback(epoch_end_callback)], verbose=verbose)
    finally:
      training_lock.release()
    if training_finished_callback:
      plt.subplot(1, 2, 1)
      plt.plot(history.history['accuracy'])
      plt.title('model accuracy')
      plt.ylabel('accuracy')
      plt.xlabel('epoch')
      plt.legend(['train'], loc='upper left')
      plt.subplot(1, 2, 2)
      plt.plot(history.history['loss'])
      plt.title('model loss')
      plt.ylabel('loss')
      plt.xlabel('epoch')
      plt.legend(['train'], loc='upper left')
      f = io.BytesIO()
      plt.savefig(f, format="PNG")
      plt.close()
      f.seek(0)
      training_finished_callback(f)
  Thread(target=inner).start()
  return True

# %%
# парсим текст, как последовательность символов
num_characters = 34  # 33 буквы + пробел
# токенизируем на уровне символов
tokenizer = Tokenizer(num_words=num_characters, char_level=True)
# формируем токены на основе частотности внашем тексте
tokenizer.fit_on_texts([text])
print(tokenizer.word_index)
inp_chars = 6
model = Sequential()
# при тренировке в рекуррентные модели keras подается сразу вся последовательность, поэтому в input теперь два числа. 1-длина последовательности, 2-размер OHE
model.add(Input((inp_chars, num_characters)))
# рекуррентный слой на 500 нейронов
model.add(SimpleRNN(128, activation='tanh'))
model.add(Dense(num_characters, activation='softmax'))
model.summary()
train_model(text, epochs=1)

# %%
@timeit
def buildPhrase(inp_str, str_len=50):
  if not training_lock.acquire(blocking=False):
    return "Training in progress"
  try:
    for i in range(str_len):
      x = tokenizer.texts_to_matrix(inp_str[i:i+inp_chars])
      inp = x.reshape(1, inp_chars, num_characters)
      pred = model(inp, training=False)  # предсказываем OHE четвертого символа
      # получаем ответвет символьном представлении
      d = tokenizer.index_word[pred.numpy().argmax(axis=1)[0]]
      inp_str += d  # дописываем строку
  finally:
    training_lock.release()
  return inp_str
res = buildPhrase("утренн")
print(res)

# %%
class TrainingInfoSingleton:
  current_epoch: int = 0
  total_epochs: int = -1
  result_graph: None|io.BytesIO = None
  training_loss: str = "N/A"
  training_accuracy: str = "N/A"

class RequestHandler(BaseHTTPRequestHandler):
  def __init__(self, *args, training_info: None|TrainingInfoSingleton = None, **kwargs):
    if training_info is None:
      raise Exception("training_info must be passed")
    self.training_info: TrainingInfoSingleton = training_info
    super().__init__(*args, **kwargs)

  def do_POST(self):
    if self.path not in ["/ajax/train-model"]:
      return self.send_response_only(HTTPStatus.NOT_FOUND)
    ctype, pdict = cgi.parse_header(self.headers.get('content-type'))
    try:
      if ctype == 'multipart/form-data':
        postvars = cgi.parse_multipart(self.rfile, pdict)
      elif ctype == 'application/x-www-form-urlencoded':
        length = int(self.headers.get('content-length'))
        postvars = parse_qs(self.rfile.read(length).decode('utf-8'),
                            max_num_fields=4,
                            keep_blank_values=True)
      else:
        postvars = {}
      params = {}
      for key, _type in [("input-text", str),
                         ("epochs", int),
                         ("learning-rate", float),
                         ("batch-size", int)]:
        if key not in postvars or not postvars[key][0]:
          raise Exception("not enough arguments")
        params[key.replace('-', '_')] = _type(postvars[key][0])
        del postvars[key]
      if postvars:
        raise Exception("unknown params")

      def epoch_finished(epoch, logs):
        self.training_info.training_loss = str(logs['loss'])
        self.training_info.training_accuracy = str(logs['accuracy'])
        self.training_info.current_epoch = epoch

      def training_finished(file):
        self.training_info.result_graph = file

      if train_model(**params,
                     epoch_end_callback=epoch_finished,
                     training_finished_callback=training_finished,
                     verbose=0):
        self.training_info.current_epoch = 0
        self.training_info.total_epochs = params["epochs"]
        self.training_info.result_graph = None
        self.training_info.training_loss = "N/A"
        self.training_info.training_accuracy = "N/A"
        self.send_response(HTTPStatus.OK)
        self.send_header('Content-Length', '0')
        self.end_headers()
      else:
        self.send_response_only(HTTPStatus.INTERNAL_SERVER_ERROR)

    except Exception as e:
      self.send_response_only(HTTPStatus.BAD_REQUEST)
      print(e)
      return

  def do_GET(self):
    url = urlparse(self.path)
    if url.path not in [
      "/", 
      "/ajax/update-learning-progress", 
      "/ajax/generate-text"
      ]:
      self.send_response_only(HTTPStatus.NOT_FOUND)
      return
    query = None
    try:
      query = parse_qs(url.query,
                       max_num_fields=2,
                       keep_blank_values=True)
    except Exception as e:
      self.send_response_only(HTTPStatus.BAD_REQUEST)
      print(e)
      return
    print(query)
    match url.path:
      case "/" if not query:  
        return self.process_new_page()
      case "/ajax/update-learning-progress" if not query:
        return self.process_update_training_progress()
      case "/ajax/generate-text":
        return self.process_generate_text_request(query)
      case _:      
        self.send_response_only(HTTPStatus.BAD_REQUEST)
        return

  def process_update_training_progress(self):
    enc = sys.getfilesystemencoding()
    params = {}
    if self.training_info.total_epochs == -1:
      self.send_response_only(HTTPStatus.NO_CONTENT)
    if self.training_info.result_graph:
      params["still_learning"] = False
      result = self.training_info.result_graph.getvalue()
      params["plots"] = "data:image/png;base64, "+base64.b64encode(result).decode('ascii')
      TOTAL_LENGTH = 80
      complete_amount = 80
      current_epoch = self.training_info.total_epochs
    else:
      TOTAL_LENGTH = 80
      complete_amount = (self.training_info.current_epoch*TOTAL_LENGTH//self.training_info.total_epochs)
      params["still_learning"] = True
      current_epoch = self.training_info.current_epoch
    filled = "="*complete_amount
    arrow = ">" if complete_amount != TOTAL_LENGTH else ""
    unfilled = "&nbsp;"*(TOTAL_LENGTH-complete_amount-len(arrow))
    loss = self.training_info.training_loss
    accuracy = self.training_info.training_accuracy
    params["learning_progress"] = f"{current_epoch}/{self.training_info.total_epochs} [{filled}{arrow}{unfilled}] {loss=}, {accuracy=}"

    encoded = json.dumps(params).encode(enc, 'surrogateescape')
    f = io.BytesIO()
    f.write(encoded)
    f.seek(0)
    self.send_response(HTTPStatus.OK)
    self.send_header(f"Content-type", "application/json; charset={enc}")
    self.send_header("Content-Length", str(len(encoded)))
    self.end_headers()

    self.copyfile(f, self.wfile)

  def process_generate_text_request(self, query):
    enc = sys.getfilesystemencoding()
    if "text-prefix" not in query and "target-length" not in query:
      print("Invalid query")
      self.send_response_only(HTTPStatus.BAD_REQUEST)
      return

    result = ""
    spent_time = 0
    try:
      text_prefix: str = query["text-prefix"][0]
      # убираем первый невидимый символ
      text = text_prefix.replace('\ufeff', '')

      # заменяем все символы кроме кириллицы на пустые символы
      text = re.sub(r'[^А-я ]', '', text)
      target_length = int(query["target-length"][0])
      spent_time, result = buildPhrase(text, target_length)
    except Exception as e:
      print(e)
      self.send_response_only(HTTPStatus.INTERNAL_SERVER_ERROR)
      return
    result_dict = {"generated_text": result, "time": spent_time}
    result_json = json.dumps(result_dict)
    f = io.BytesIO()
    f.write(result_json.encode(enc, 'surrogateescape'))
    f.seek(0)

    self.send_response(HTTPStatus.OK)
    self.send_header(f"Content-type", "application/json; charset={enc}")
    self.send_header("Content-Length", str(len(result_json)))
    self.end_headers()

    self.copyfile(f, self.wfile)

  def process_new_page(self):
    enc = sys.getfilesystemencoding()
    title = 'Recurrent neural network for text generation example'
    template = ""
    with open('template.html', 'r', encoding='utf-8') as template_file:
      template = template_file.read()
    template = template.replace('{{title}}', title).replace('{{enc}}', enc)
    encoded = template.encode(enc, 'surrogateescape')
    f = io.BytesIO()
    f.write(encoded)
    f.seek(0)
    self.send_response(HTTPStatus.OK)
    self.send_header(f"Content-type", "text/html; charset={enc}")
    self.send_header("Content-Length", str(len(encoded)))
    self.end_headers()

    self.copyfile(f, self.wfile)

  def copyfile(self, source, outputfile):
    shutil.copyfileobj(source, outputfile)

tis = TrainingInfoSingleton()
def create_request_handler(*args, **kwargs):
  return RequestHandler(*args, **kwargs, training_info=tis)

serv = HTTPServer(('', 80), create_request_handler, True)
while True:
  try:
    serv.handle_request()
  except KeyboardInterrupt:
    print("Interrupted")
    break