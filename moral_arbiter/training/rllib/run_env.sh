cd ~/moral-arbiter/moral_arbiter/training/rllib/


echo "AI?"
select yn in "Yes_phase1" "Yes_phase2" "No"; do
  case $yn in
    Yes_phase1 ) python training_script.py --run-dir envs/AI/layout/phase1/ > envs/AI/layout/phase1/logfile.txt 2>&1&;;
    Yes_phase2 ) python training_script.py --run-dir envs/AI/layout/phase2/ > envs/AI/layout/phase2/logfile.txt 2>&1&;;
    No ) exit;;
  esac
  break;
done


echo "Amoral?"
select yn in "Yes_layout" "Yes_rand" "No"; do
  case $yn in
    Yes_layout ) python training_script.py --run-dir envs/amoral/layout/ > envs/amoral/layout/logfile.txt 2>&1&;;
    Yes_rand ) python training_script.py --run-dir envs/amoral/random/ > envs/amoral/random/logfile.txt 2>&1&;;
    No ) exit;;
  esac
  break;
done

echo "Virtue Ethics?"
select yn in "Yes_layout" "Yes_rand" "No"; do
  case $yn in
    Yes_layout ) python training_script.py --run-dir envs/virtue_ethics/layout/ >envs/virtue_ethics/layout/logfile.txt 2>&1&;;
    Yes_rand ) python training_script.py --run-dir envs/virtue_ethics/random/ > envs/virtue_ethics/random/logfile.txt 2>&1&;;
    No ) exit;;
  esac
  break;
done

echo "utilitarian?"
select yn in "Yes_layout" "Yes_rand" "No"; do
  case $yn in
    Yes_layout ) python training_script.py --run-dir envs/utilitarian/layout/ > envs/utilitarian/layout/logfile.txt 2>&1&;;
    Yes_rand ) python training_script.py --run-dir envs/utilitarian/random/ > envs/utilitarian/random/logfile.txt 2>&1&;;
    No ) exit;;
  esac
  break;
done

echo "default?"
select yn in "Yes" "No"; do
  case $yn in
    Yes ) python training_script.py --run-dir envs/og/phase1/ > envs/og/phase1/logfile.txt 2>&1&;;
    No ) exit;;
  esac
  break;
done