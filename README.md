# UMA-Predict
multimodel assisted quantitative horse racing predictor
## Project Structure
```
horse_quant/
│
├── data/                    
│   ├── raw/           
│   │   ├── races.csv
│   │   ├── horses.csv
│   │   ├── jockeys.csv
│   │   └── news.csv
│   ├── processed/         
│   │   ├── horse_sequences.parquet
│   │   └── race_features.parquet
│   └── payout/             
│       └── win_payoff.parquet
│
├── features/      
│   ├── alpha/               
│   ├── pace/          
│   ├── pedigree/           
│   └── jockey_trainer/     
│
├── dataset/                 
│   ├── horse_sequence_dataset.py 
│   └── data_handler.py           
│
├── models/                   
│   ├── transformer/         
│   ├── tft/                  
│   ├── gat/                 
│   └── CB_baseline/         
│
├── multimodal/               
│   ├── text_encoder.py  
│   └── cross_attention.py   
│
├── backtest/               
│   ├── roi_calculator.py
│   └── simulated_betting.py
│
├── workflow/                
│   ├── train_pipeline.py    
│   └── eval_pipeline.py     
│
└── scripts/                 
    ├── build_sequences.py   
    ├── fetch_data.py       
    └── feature_gen.py     
```
## Temporary Model Structure
```
┌───────────────────────────┐
│ Input: Horse Historical Seq│
│ Features:                  │
│ - Past race results        │
│ - Pace / Last3f / Weight   │
│ - Jockey / Trainer stats   │
└───────────────┬───────────┘
                │
          Temporal Encoder
      (Transformer / TFT / LSTM)
                │
┌───────────────┴───────────────┐
│ Race Interaction Encoder      │
│ - Self-attention on all horses│
│ - GAT                         │
└───────────────┬───────────────┘
                │
        Optional Cross Attention
        (with News / Comments)
                │
          Output Layer
        - Softmax Win Prob
        - Top-3 Probabilities
                │
          Loss Function
        - ListMLE / LambdaRank
                │
          Post-Processing
        - Compare with Odds
        - Expected Value / ROI
```