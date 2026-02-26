# OCR GOT2.0
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. convert model
```bash
optimum-cli export openvino --model stepfun-ai/GOT-OCR-2.0-hf --weight-format int4 GOT-OCR-2.0-hf\INT4
```

3. run 
```bash
python run.py
```