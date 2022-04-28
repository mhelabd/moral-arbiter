cd ~/ai-ethicist/ai_economist/training/rllib/
for config_file in $(ls  -d envs/utilitarian/layout/agent*);
do
    echo "starting... $config_file"
    python training_script.py --run-dir ${config_file} > ${config_file}/logfile.txt 2>&1&
    while ! tail -1 ${config_file}/logfile.txt | grep "Final snapshot saved! All done." > /dev/null;
    do
        sleep 5;
    done
    echo "$config_file now done!"
done