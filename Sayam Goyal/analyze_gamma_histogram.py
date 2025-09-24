import argparse, json, os, time
from datetime import datetime, timedelta
import requests
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

GAMMA = "https://gamma-api.polymarket.com/"
CLOB = "https://clob.polymarket.com/"
CACHE_DIR = "data/gamma_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_markets_since(months=2, limit=5000):
    now = datetime.utcnow()
    start_date = (now - timedelta(days=30 * months)).strftime("%Y-%m-%d")
    params = {"order":"startDate","start_date_min":start_date,"closed":"true","limit":str(limit),"offset":"0"}
    cf = os.path.join(CACHE_DIR, f"markets_{months}m.json")
    if os.path.exists(cf):
        with open(cf, "r") as f: return json.load(f)
    allm=[]; off=0
    pbar=tqdm(desc="Fetching markets", unit="page")
    while True:
        params["offset"] = str(off)
        r = requests.get(GAMMA+"markets", params=params, timeout=30); r.raise_for_status()
        data = r.json();
        if not data: break
        allm.extend(data); off += int(params["limit"]); pbar.update(1); time.sleep(0.2)
    pbar.close()
    with open(cf, "w") as f: json.dump(allm,f)
    return allm

def is_sports_or_crypto(m):
    t = m.get("name","").lower()
    if m.get("gameStartTime") is not None: return True
    tags = m.get("tags") or m.get("related_tags")
    if isinstance(tags,str):
        try: tags_list = json.loads(tags)
        except: tags_list=[tags]
    else: tags_list = tags or []
    tt = " ".join([str(x).lower() for x in tags_list])
    kw = ["sport","nba","nfl","football","crypto","bitcoin","btc","ethereum","eth"]
    return any(k in t or k in tt for k in kw)

def get_yes_outcome(m):
    of = m.get("outcomes")
    if not of: return None
    try: outs = json.loads(of)
    except: outs = of
    if not isinstance(outs,list): return None
    for i,o in enumerate(outs):
        if isinstance(o,str) and o.lower()=="yes": return i
    return 0

def fetch_yes_price_12h_before(token, close_ts):
    cf = os.path.join(CACHE_DIR, f"prices_{token}.json")
    if os.path.exists(cf):
        with open(cf,"r") as f: hist = json.load(f)
    else:
        s = int(max(0, close_ts - 60*60*48)); e = int(close_ts)
        params={"market":token,"startTs":s,"endTs":e,"fidelity":"60"}
        r = requests.get(CLOB+"prices-history", params=params, timeout=30); r.raise_for_status(); hist = r.json().get("history",[])
        with open(cf,"w") as f: json.dump(hist,f)
        time.sleep(0.2)
    if not hist: return None
    tgt = close_ts - 12*3600; chosen=None
    for h in hist:
        try: t=int(h.get("t"))
        except: continue
        if t<=tgt: chosen=h
        else: break
    if chosen is None: chosen=hist[0]
    try: return float(chosen.get("p"))
    except: return None

def analyze(markets):
    recs=[]
    for m in tqdm(markets, desc="Processing markets"):
        try:
            if is_sports_or_crypto(m): continue
            ct = m.get("clobTokenIds")
            if not ct: continue
            try: tokens = json.loads(ct)
            except: continue
            if not isinstance(tokens,list) or len(tokens)<2: continue
            yi = get_yes_outcome(m)
            if yi is None: continue
            closed = m.get("closedTime")
            if not closed: continue
            cc = closed.split(".")[0].replace("+00",
            "")
            try: dt = datetime.fromisoformat(cc)
            except:
                try: dt = datetime.strptime(cc, "%Y-%m-%dT%H:%M:%S")
                except: continue
            cts = int(dt.timestamp())
            token = tokens[yi]
            p12 = fetch_yes_price_12h_before(token, cts)
            if p12 is None: continue
            opf = m.get("outcomePrices")
            if not opf: continue
            try: ops = json.loads(opf)
            except: continue
            winner = next((i for i,v in enumerate(ops) if str(v)=="1"), None)
            if winner is None: continue
            yes_won = 1 if winner==yi else 0
            recs.append({"market_id":m.get("id"),"name":m.get("name"),"yes_price_12h":p12,"yes_won":yes_won})
        except: continue
    df = pd.DataFrame(recs)
    if df.empty: print("No data"); return
    bins=[i/20.0 for i in range(21)]; labels=[f"{int(b*100)}-{int((b+0.05)*100)}%" for b in bins[:-1]]
    df["bucket"] = pd.cut(df["yes_price_12h"], bins=bins, labels=labels, include_lowest=True)
    summary = df.groupby("bucket").agg(predicted_mean=("yes_price_12h","mean"), actual_fraction=("yes_won","mean"), count=("yes_won","count")).reset_index()
    x=list(range(len(summary))); w=0.35
    fig,ax=plt.subplots(figsize=(12,6))
    ax.bar([i-w/2 for i in x], summary["predicted_mean"], width=w, label="Predicted")
    ax.bar([i+w/2 for i in x], summary["actual_fraction"], width=w, label="Actual")
    ax.set_xticks(x); ax.set_xticklabels(summary["bucket"], rotation=45)
    for i,row in summary.iterrows():
        cnt = int(row["count"]) if not pd.isna(row["count"]) else 0
        ax.text(i-w/2, (row.predicted_mean or 0)+0.01, f"n={cnt}", ha="center", fontsize=8)
    plt.tight_layout(); out=os.path.join(CACHE_DIR,"predicted_vs_actual_histogram.png"); plt.savefig(out); print(out)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--months",type=int,default=2); args=p.parse_args()
    ms = get_markets_since(months=args.months)
    print(f"Fetched {len(ms)} markets")
    analyze(ms)

if __name__=="__main__": main()
