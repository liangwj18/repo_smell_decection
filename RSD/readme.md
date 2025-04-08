1.You need to download dataset in (https://pan.baidu.com/s/1u5bVmRTh_AZhHUvXO3cf6A |Keyword:g8j6) unzip it, and put it into RSD/RSD_dataset (Required)
    example:RSD/RSD_dataset/binary_classification/Broken Hierarchy_test.jsonl
2.You could download pretreatment in (https://pan.baidu.com/s/1ETL2SwWnGw-U9YNGaPmOYg |Keyword:39af)  unzip it(Recomended), and put it into RSD/RSD_pretreatment, or you need to download distilbert-base-uncased and put it in to RSD/model/distilbert-base-uncased and wait a few time.
    example:RSD/RSD_pretreatment/binary_classification_Broken Hierarchy_test/cls.pth

3.You could download original github repository in (https://pan.baidu.com/s/1_E1cl4sBcEvBkpzYWni5gQ |Keyword:jreb)(Optional)

4.You can run the training program:  In the repo_smell_upstream directory, 
    setting RSD/train_2.py's binary_train()'s benchmark = 'train'
    python -m RSD.train_2
The default experimental mode is Full. If you want to perform an ablation experiment, you can change the annotated part of the AttentionMapClassifier in the network.

5.You can run the testing program:   In the repo_smell_upstream directory, 
    setting RSD/train_2.py's binary_train()'s benchmark = 'test'
    python -m RSD.train_2

6.You can see the testing metric of f1,acc, recall:   In the repo_smell_upstream directory, 
  
    python -m RSD.generate_csv_7
