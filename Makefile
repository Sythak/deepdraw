####################### PACKAGE ACTION ############################

reinstall_package:
	@pip uninstall -y deep_draw || :
	@pip install -e .

run_preprocess_train_eval:
	python -c 'from deep_draw.interface.main import preprocess_train_eval; preprocess_train_eval()'

run_pred:
	python -c 'from deep_draw.interface.main import pred; pred()'

run streamlit:
	python -c 'from deep_draw.interface' streamlit run app.python
