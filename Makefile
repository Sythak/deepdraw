####################### PACKAGE ACTION ############################

reinstall_package:
	@pip uninstall -y deep_draw || :
	@pip install -e .
