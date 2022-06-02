# python3 train_model.py --experiment Test --model NN --validation_year 2020 --shift 0 --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 1
# python3 train_model.py --experiment Test --model NN --validation_year 2020 --shift 4320 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 --dropout 0.5 --epochs 1

python3 train_model.py --experiment Forecaster-0 --model NN --validation_year 2020 --shift 0  --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20
python3 train_model.py --experiment Forecaster-24 --model NN --validation_year 2020 --shift 24 --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20
python3 train_model.py --experiment Forecaster-168 --model NN --validation_year 2020 --shift 168 --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20
python3 train_model.py --experiment Forecaster-720 --model NN --validation_year 2020 --shift 720 --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20
python3 train_model.py --experiment Forecaster-4320 --model NN --validation_year 2020 --shift 4320 --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20

python3 train_model.py --experiment Forecaster-0 --model NN --validation_year 2020 --shift 0 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20
python3 train_model.py --experiment Forecaster-24 --model NN --validation_year 2020 --shift 24 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20
python3 train_model.py --experiment Forecaster-168 --model NN --validation_year 2020 --shift 168 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20
python3 train_model.py --experiment Forecaster-720 --model NN --validation_year 2020 --shift 720 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20
python3 train_model.py --experiment Forecaster-4320 --model NN --validation_year 2020 --shift 4320 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d --dropout 0.75 --epochs 20

python3 train_model.py --experiment Forecaster-0 --model NN --validation_year 2020 --shift 0 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 --dropout 0.5 --epochs 20
python3 train_model.py --experiment Forecaster-24 --model NN --validation_year 2020 --shift 24 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 --dropout 0.5 --epochs 20
python3 train_model.py --experiment Forecaster-168 --model NN --validation_year 2020 --shift 168 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 --dropout 0.5 --epochs 20
python3 train_model.py --experiment Forecaster-720 --model NN --validation_year 2020 --shift 720 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 --dropout 0.5 --epochs 20
python3 train_model.py --experiment Forecaster-4320 --model NN --validation_year 2020 --shift 4320 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 --dropout 0.5 --epochs 20

python3 train_gimlinear_model.py --experiment GIMLinear-0 --validation_year 2020 --shift 0  --periods 24h 27d 180d 366d 4020d 
python3 train_gimlinear_model.py --experiment GIMLinear-24 --validation_year 2020 --shift 24 --periods 24h 27d 180d 366d 4020d
python3 train_gimlinear_model.py --experiment GIMLinear-168 --validation_year 2020 --shift 168 --periods 24h 27d 180d 366d 4020d 
python3 train_gimlinear_model.py --experiment GIMLinear-720 --validation_year 2020 --shift 720 --periods 24h 27d 180d 366d 4020d 
python3 train_gimlinear_model.py --experiment GIMLinear-4320 --validation_year 2020 --shift 4320 --periods 24h 27d 180d 366d 4020d 

python3 train_gimlinear_model.py --experiment GIMLinear-0 --validation_year 2020 --shift 0 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d 
python3 train_gimlinear_model.py --experiment GIMLinear-24 --validation_year 2020 --shift 24 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d
python3 train_gimlinear_model.py --experiment GIMLinear-168 --validation_year 2020 --shift 168 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d 
python3 train_gimlinear_model.py --experiment GIMLinear-720 --validation_year 2020 --shift 720 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d 
python3 train_gimlinear_model.py --experiment GIMLinear-4320 --validation_year 2020 --shift 4320 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d 

python3 train_gimlinear_model.py --experiment GIMLinear-0 --validation_year 2020 --shift 0 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 
python3 train_gimlinear_model.py --experiment GIMLinear-24 --validation_year 2020 --shift 24 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 
python3 train_gimlinear_model.py --experiment GIMLinear-168 --validation_year 2020 --shift 168 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 
python3 train_gimlinear_model.py --experiment GIMLinear-720 --validation_year 2020 --shift 720 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 
python3 train_gimlinear_model.py --experiment GIMLinear-4320 --validation_year 2020 --shift 4320 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320

python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-0 --validation_year 2020 --shift 0  --periods 24h 27d 180d 366d 4020d 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-24 --validation_year 2020 --shift 24 --periods 24h 27d 180d 366d 4020d
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-168 --validation_year 2020 --shift 168 --periods 24h 27d 180d 366d 4020d 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-720 --validation_year 2020 --shift 720 --periods 24h 27d 180d 366d 4020d 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-4320 --validation_year 2020 --shift 4320 --periods 24h 27d 180d 366d 4020d 

python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-0 --validation_year 2020 --shift 0 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-24 --validation_year 2020 --shift 24 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-168 --validation_year 2020 --shift 168 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-720 --validation_year 2020 --shift 720 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-4320 --validation_year 2020 --shift 4320 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020d 

python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-0 --validation_year 2020 --shift 0 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-24 --validation_year 2020 --shift 24 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-168 --validation_year 2020 --shift 168 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-720 --validation_year 2020 --shift 720 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320 
python3 train_gimlixgbdt_model.py --experiment GIMLi-XGBDT-4320 --validation_year 2020 --shift 4320 --indices f10.7 --log_indices --sqrt_indices --periods 24h 27d 180d 366d 4020 --lookback 24 --lookback_windowed 648 4320
