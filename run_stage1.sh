export EPOCH="200" # change to your gpu id



# for name in youtube3 youtube5 youtube_test youtube_bee #youtube2 youtube3 youtube5 youtube_test youtube_bee sora zju_377 zju_386 zju_387 zju_392 zju_393 zju_394  
# do              
#     echo "output_${name}/${name}_stage1 "
#     python train.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type wild --train_stage 1 --epochs 200 --pose_op_start_iter 10 
#     python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type wild --epoch ${EPOCH} --pose observ --stage 1
#     python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type wild --epoch ${EPOCH} --pose tpose --stage 1


# done

# for name in zju_377 zju_386 zju_392  
# do
#     echo "output_${name}/${name}_stage1"
#     python train.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type zju_mocap --train_stage 1 --epochs 200 --pose_op_start_iter 10 
#     python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type zju_mocap --epoch ${EPOCH} --pose observ --stage 1
#     python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type zju_mocap --epoch ${EPOCH} --pose tpose --stage 1

# done

# for name in zju_387 zju_393 zju_394  
# do
#     echo "output_${name}/${name}_stage1"
#     python train.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type zju_mocap_1 --train_stage 1 --epochs 200 --pose_op_start_iter 10 
#     python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type zju_mocap_1 --epoch ${EPOCH} --pose observ --stage 1
#     python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type zju_mocap_1 --epoch ${EPOCH} --pose tpose --stage 1

# done

for name in monocap_olek monocap_vlad #monocap_olek monocap_vlad monocap_lan monocap_marc 
do                            
    echo "output_${name}/${name}_stage1"
    python train.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type monocap --train_stage 1 --epochs 200 --pose_op_start_iter 10 
    python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type monocap --epoch ${EPOCH} --pose observ --stage 1
    python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_stage1 --src_type monocap --epoch ${EPOCH} --pose tpose --stage 1

done




