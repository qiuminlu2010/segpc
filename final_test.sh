python tools/ensemble_test.py \
    --config ./best_models/test/model1_for_cell.py ./best_models/test/model2_for_cell.py \
    --checkpoint ./best_models/test/model1_for_cell_best.pth ./best_models/test/model2_for_cell_best.pth \
    --out ./pred/ensemble_cell_test.pkl  

python tools/test.py \
    ./best_models/test/model1_for_nu.py \
    ./best_models/test/model1_for_nu_best.pth \
    --out ./pred/model1_nu_test.pkl 

