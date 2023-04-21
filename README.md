# CS6220 Project

This repository is for 2022-2023 semester 2 CS6220 project named **What does a pre-trained climatology language model learn?**

You can try the code with `python train3.py
    --data_dir ${GLUE_DIR}
    --job_id ${JOB_ID}
    --warmup_proportion 0.1
    --num_train_epochs 50
    --eval_step 20
    --output_dir ${OUTPUT_DIR}/${TASK_NAME}/${JOB_ID}
    --task_name $TASK_NAME
    --weight_decay 0.01
    --batch_size 720
    --learning_rate 0.0001`
