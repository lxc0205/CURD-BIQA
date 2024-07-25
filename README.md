CURD-BIQA

## Dependencies

```
conda create -n curd python=3.8.18
conda activate curdbiqa
pip install -r requirements.txt
```

## Usage
### Multiscale framework for BIQA
```
python framework.py <--curd> --method <method> --dataset <dataset> --index <indexs> --beta <betas>
```

Some available options:
* `--dataset`: Testing dataset, support datasets:  koniq-10k | live | csiq | tid2013.
* `--method`: basic method.
* `--curd`: The flag of using curd (Using multiscale framework), switch off represents the original method.
* `--index`: The curd selected indexs, only predicting process used.
* `--beta`: The curd selected betas, only predicting process used.


Outputs:
* `orignal mehthod and predicting processing`: print `PLCC` and `SRCC`
* `Usage of curd, get the layer scores`: .\outputs\\<method\>\ multiscale outputs\\\<dataset>.txt

### CURD-BIQA
```
python curd_iqa.py --method <method> --inputFileSet <input file names> --outputFile <output file name> --save_num <save_num>
```

Some available options:
* `--method`: basic method.
* `--inputFileSet`: input file names in multiscale outputs folder, support names:  koniq-10k.txt | live.txt | csiq.txt | tid2013.txt.
* `--outputFile`: output file names.
* `--save_num`: Save numbers, default number is 50000.

Outputs:
* `output file`: .\outputs\\<method\>\curd outputs\\\<outputFile>.txt