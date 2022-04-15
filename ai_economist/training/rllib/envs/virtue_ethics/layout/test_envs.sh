cd ~/ai-ethicist/ai_economist/training/rllib/
for config_file in $(ls  -d envs/virtue_ethics/layout/agent*);
do
    python training_script.py --run-dir ${config_file}/predefined_skill > ${config_file}/predefined_skill/logfile.txt 2>&1&
    while ! tail -1 ${config_file}/predefined_skill/logfile.txt | grep "Final snapshot saved! All done." > /dev/null;
    do
        sleep 5;
    done
done |  tqdm --total $(ls  -d envs/virtue_ethics/layout/agent* | wc -l)  >> /dev/null