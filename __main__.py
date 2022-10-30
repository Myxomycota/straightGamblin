import requests
import pandas as pd
import sqlite3
import shutil
import os

# %%
if(os.path.isdir("/work/cardiffnlp")):
    shutil.rmtree("/work/cardiffnlp")
def main(args):
  # %%
  from transformers import AutoModelForSequenceClassification
  from transformers import TFAutoModelForSequenceClassification
  from transformers import AutoTokenizer
  import numpy as np
  from scipy.special import softmax
  import csv
  import urllib.request

  # Preprocess text (username and link placeholders)
  def preprocess(text):
      new_text = []


      for t in text.split(" "):
          t = '@user' if t.startswith('@') and len(t) > 1 else t
          t = 'http' if t.startswith('http') else t
          new_text.append(t)
      return " ".join(new_text)

  # Tasks:
  # emoji, emotion, hate, irony, offensive, sentiment
  # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

  task='sentiment'
  MODEL = f"cardiffnlp/twitter-roberta-base-{task}"

  tokenizer = AutoTokenizer.from_pretrained(MODEL)

  # download label mapping
  labels=[]
  mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
  with urllib.request.urlopen(mapping_link) as f:
      html = f.read().decode('utf-8').split("\n")
      csvreader = csv.reader(html, delimiter='\t')
  labels = [row[1] for row in csvreader if len(row) > 1]

  # PT
  model = AutoModelForSequenceClassification.from_pretrained(MODEL)
  # model.save_pretrained(MODEL)

  text = "Good night 😊"
  text = preprocess(text)
  encoded_input = tokenizer(text, return_tensors='pt')
  output = model(**encoded_input)
  scores = output[0][0].detach().numpy()
  scores = softmax(scores)

  # # TF
  # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
  # model.save_pretrained(MODEL)

  # text = "Good night 😊"
  # encoded_input = tokenizer(text, return_tensors='tf')
  # output = model(encoded_input)
  # scores = output[0][0].numpy()
  # scores = softmax(scores)

  ranking = np.argsort(scores)
  ranking = ranking[::-1]
  for i in range(scores.shape[0]):
      l = labels[ranking[i]]
      s = scores[ranking[i]]
      print(f"{i+1}) {l} {np.round(float(s), 4)}")


  # %%
  # note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
  auth = requests.auth.HTTPBasicAuth('gcc-_j7T5tHsrMSW7QlmHw', 'B_ahlLsbY6r3UHbQo_pDtiRPiPqlzw')

  # %%
  # here we pass our login method (password), username, and password
  data = {'grant_type': 'password',
          'username': 'fomo_erotic',
          'password': 'Msr=1999'}

  # %%
  # setup our header info, which gives reddit a brief description of our app
  headers = {'User-Agent': 'MyBot/0.0.1'}

  # %%
  # send our request for an OAuth token
  res = requests.post('https://www.reddit.com/api/v1/access_token',
                      auth=auth, data=data, headers=headers)

  # %%
  # convert response to JSON and pull access_token value
  TOKEN = res.json()['access_token']

  # %%
  conn = sqlite3.connect('/work/output.sql')

  # %%
  last_dt = pd.read_sql_query("SELECT * FROM daily_sentiment LIMIT 1", conn)

  # %%
  # add authorization to our headers dictionary
  headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}
  params = {'before':last_dt['name'][0],'limit':1000}

  # %%
  # while the token is valid (~2 hours) we just add headers=headers to our requests
  requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)

  # %%
  res = requests.get("https://oauth.reddit.com/r/wallstreetbets/hot",headers=headers)

  # %%
  d_thread = res.json()['data']['children'][0]['data']['permalink']

  # %%
  comment = 1
  thread = requests.get("https://oauth.reddit.com/"+d_thread,
                     headers=headers,params=params)
  print(thread.json()[1]['data']['children'][comment]['data']['author'])
  print(thread.json()[1]['data']['children'][comment]['data']['body'])
  # thread.json()[1]['data']['children'][comment]['data']

  # %%
  df = pd.DataFrame()  # initialize dataframe

  # loop through each post retrieved from GET request
  for comment in thread.json()[1]['data']['children']:
      # append relevant data to dataframe
      if comment['kind'] != 'more':

        text = preprocess(comment['data']['body'])
        encoded_input = tokenizer(text[:512], return_tensors='pt')
        output = model(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        df = df.append({
            'created_utc': comment['data']['created_utc'],
            'parent_id': comment['data']['parent_id'],
            'name': comment['data']['name'],
            'author': comment['data']['author'],
            'comment': comment['data']['body'],
            'positive': scores[2],
            'neutral': scores[1],
            'negative': scores[0],
            'SPY': ('SPY' in comment['data']['body']) or ('spy' in comment['data']['body']),
            'Bull': ('Bull' in comment['data']['body']) or ('bull' in comment['data']['body']),
            'Bear': ('Bear' in comment['data']['body']) or ('bear' in comment['data']['body']),
            'Call': ('Call' in comment['data']['body']) or ('call' in comment['data']['body']),
            'Put': ('Put' in comment['data']['body']) or ('put' in comment['data']['body'])
          }, ignore_index=True)

  # %%
  df.to_sql('daily_sentiment', conn, if_exists='append', index=False)

  # %%
  """
  <a style='text-decoration:none;line-height:16px;display:flex;color:#5B5B62;padding:10px;justify-content:end;' href='https://deepnote.com?utm_source=created-in-deepnote-cell&projectId=41df91d8-946a-40e7-a6a7-bad56c01e85a' target="_blank">
  <img alt='Created in deepnote.com' style='display:inline;max-height:16px;margin:0px;margin-right:7.5px;' src='data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0iVVRGLTgiPz4KPHN2ZyB3aWR0aD0iODBweCIgaGVpZ2h0PSI4MHB4IiB2aWV3Qm94PSIwIDAgODAgODAiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayI+CiAgICA8IS0tIEdlbmVyYXRvcjogU2tldGNoIDU0LjEgKDc2NDkwKSAtIGh0dHBzOi8vc2tldGNoYXBwLmNvbSAtLT4KICAgIDx0aXRsZT5Hcm91cCAzPC90aXRsZT4KICAgIDxkZXNjPkNyZWF0ZWQgd2l0aCBTa2V0Y2guPC9kZXNjPgogICAgPGcgaWQ9IkxhbmRpbmciIHN0cm9rZT0ibm9uZSIgc3Ryb2tlLXdpZHRoPSIxIiBmaWxsPSJub25lIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiPgogICAgICAgIDxnIGlkPSJBcnRib2FyZCIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoLTEyMzUuMDAwMDAwLCAtNzkuMDAwMDAwKSI+CiAgICAgICAgICAgIDxnIGlkPSJHcm91cC0zIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjM1LjAwMDAwMCwgNzkuMDAwMDAwKSI+CiAgICAgICAgICAgICAgICA8cG9seWdvbiBpZD0iUGF0aC0yMCIgZmlsbD0iIzAyNjVCNCIgcG9pbnRzPSIyLjM3NjIzNzYyIDgwIDM4LjA0NzY2NjcgODAgNTcuODIxNzgyMiA3My44MDU3NTkyIDU3LjgyMTc4MjIgMzIuNzU5MjczOSAzOS4xNDAyMjc4IDMxLjY4MzE2ODMiPjwvcG9seWdvbj4KICAgICAgICAgICAgICAgIDxwYXRoIGQ9Ik0zNS4wMDc3MTgsODAgQzQyLjkwNjIwMDcsNzYuNDU0OTM1OCA0Ny41NjQ5MTY3LDcxLjU0MjI2NzEgNDguOTgzODY2LDY1LjI2MTk5MzkgQzUxLjExMjI4OTksNTUuODQxNTg0MiA0MS42NzcxNzk1LDQ5LjIxMjIyODQgMjUuNjIzOTg0Niw0OS4yMTIyMjg0IEMyNS40ODQ5Mjg5LDQ5LjEyNjg0NDggMjkuODI2MTI5Niw0My4yODM4MjQ4IDM4LjY0NzU4NjksMzEuNjgzMTY4MyBMNzIuODcxMjg3MSwzMi41NTQ0MjUgTDY1LjI4MDk3Myw2Ny42NzYzNDIxIEw1MS4xMTIyODk5LDc3LjM3NjE0NCBMMzUuMDA3NzE4LDgwIFoiIGlkPSJQYXRoLTIyIiBmaWxsPSIjMDAyODY4Ij48L3BhdGg+CiAgICAgICAgICAgICAgICA8cGF0aCBkPSJNMCwzNy43MzA0NDA1IEwyNy4xMTQ1MzcsMC4yNTcxMTE0MzYgQzYyLjM3MTUxMjMsLTEuOTkwNzE3MDEgODAsMTAuNTAwMzkyNyA4MCwzNy43MzA0NDA1IEM4MCw2NC45NjA0ODgyIDY0Ljc3NjUwMzgsNzkuMDUwMzQxNCAzNC4zMjk1MTEzLDgwIEM0Ny4wNTUzNDg5LDc3LjU2NzA4MDggNTMuNDE4MjY3Nyw3MC4zMTM2MTAzIDUzLjQxODI2NzcsNTguMjM5NTg4NSBDNTMuNDE4MjY3Nyw0MC4xMjg1NTU3IDM2LjMwMzk1NDQsMzcuNzMwNDQwNSAyNS4yMjc0MTcsMzcuNzMwNDQwNSBDMTcuODQzMDU4NiwzNy43MzA0NDA1IDkuNDMzOTE5NjYsMzcuNzMwNDQwNSAwLDM3LjczMDQ0MDUgWiIgaWQ9IlBhdGgtMTkiIGZpbGw9IiMzNzkzRUYiPjwvcGF0aD4KICAgICAgICAgICAgPC9nPgogICAgICAgIDwvZz4KICAgIDwvZz4KPC9zdmc+' > </img>
  Created in <span style='font-weight:600;margin-left:4px;'>Deepnote</span></a>
  """
