.  
├── data  // dataset folders  
├── hardwareSrc  // hardware resources  
│   ├── sim  // XGB hardware models  
│   └── src  // multiple decision trees hardware accelerator  
├── model  // models classified by dataset & whether multimodal is used  
├── program  
│   ├── conv.ipynb  // training CNN model  
│   ├── deepXGB.ipynb  // CNN-XGB training  
│   ├── DMNN_by_xgboost.ipynb  // algorithm using image features alone with XGB  
│   ├── helperFunction  // custom library  
│   ├── other_program  // unused historical files  
│   ├── plot_MM_results.py  // plot multimodal experiment results  
│   ├── plot_XGB_results.py  // plot deep pruning experiment results  
│   ├── XGB_on_FPGA.ipynb  // create XGB hardware model using XGB hardware tool  
│   ├── xgb_to_vhdl  // XGB to hardware tool  
└── results  // experiment results