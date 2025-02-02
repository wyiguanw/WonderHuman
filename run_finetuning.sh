export EPOCH="300" 



# for name in youtube6 #youtube3 youtube6 youtube_test youtube_bee # youtube_bee youtube_test  #youtube_test youtube2 youtube3 youtube_bee #sora     #youtube2 youtube3 youtube_test youtube_bee sora zju_377 zju_386 zju_387 zju_392 zju_393 zju_394  
# do
#     echo "output_${name}/${name}_finetuned "
#     python train.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type wild --train_stage 3 --epochs ${EPOCH} --stage1_out_path output_${name}/${name}_stage1/net/iteration_200 
#     CUDA_VISIBLE_DEVICES=1 python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type wild --epoch ${EPOCH} --pose observ --stage 2
#     python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type wild --epoch ${EPOCH} --pose tpose --stage 2

# done

# for name in zju_392 #zju_392zju_377 zju_386 zju_392 #zju_386 zju_392 zju_377
# do
#     echo "output_${name}/${name}_finetuned "
#     #python train.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type zju_mocap --train_stage 3 --epochs ${EPOCH} --stage1_out_path output_${name}/${name}_stage1/net/iteration_200 
#     CUDA_VISIBLE_DEVICES=1  python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type zju_mocap --epoch ${EPOCH} --pose observ --stage 2
#     #python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type zju_mocap --epoch ${EPOCH} --pose tpose --stage 2

# done

# for name in zju_393 #zju_394 zju_387
# do
#     echo "output_${name}/${name}_finetuned "
#     #python train.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type zju_mocap_1 --train_stage 3 --epochs ${EPOCH} --stage1_out_path output_${name}/${name}_stage1/net/iteration_200 
#     python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type zju_mocap_1 --epoch ${EPOCH} --pose observ --stage 2
#     #python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type zju_mocap_1 --epoch ${EPOCH} --pose tpose --stage 2

# done

for name in monocap_marc #monocap_marc #monocap_olek monocap_vlad monocap_lan monocap_marc 
do
    echo "output_${name}/${name}_finetuned "
    python train.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type monocap --train_stage 3 --epochs ${EPOCH} --stage1_out_path output_${name}/${name}_stage1/net/iteration_200 
    CUDA_VISIBLE_DEVICES=1 python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type monocap --epoch ${EPOCH} --pose observ --stage 2
    python render_novel_pose.py -s ./data/${name} -m output_${name}/${name}_finetuned --src_type monocap --epoch ${EPOCH} --pose tpose --stage 2

done


