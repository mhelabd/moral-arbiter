cd ~/ai-ethicist/ai_economist/training/rllib/

echo "Virtue Ethics?"
select yn in "Yes" "No"; do
  case $yn in
    Yes ) bash ./envs/virtue_ethics/layout/run_envs.sh;;
    No ) exit;;
  esac
  break;
done

echo "utilitarian?"
select yn in "Yes" "No"; do
  case $yn in
    Yes ) bash ./envs/utilitarian/layout/run_envs.sh;;
    No ) exit;;
  esac
  break;
done

echo "AI?"
select yn in "Yes" "No"; do
  case $yn in
    Yes ) bash ./envs/AI/layout/run_envs.sh;;
    No ) exit;;
  esac
  break;
done
