for i in "607" "281" "437" "637" "22"
do
	echo $i
	CUDA_VISIBLE_DEVICES=6 python3 o2m_run.py --class_index $i --train True --evaluate True
done
