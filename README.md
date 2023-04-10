# resampling_heartsteps
Repo for "Did we personalize? Assessing personalization by an online reinforcement learning algorithm using resampling"

Given access to HeartSteps V2/V3 data,
- Install the environment.yml file.
- Create './output'
- Setup './init' with setup files from 'setup_parameters.py'
- Example Execution of Resampling Script:
```python main.py -u 0 -b 1 -s 3 -userBIdx 1 -bi Zero```
where we resample user 0 (-u 0), the second time (-b 1, -userBIdx 1), seed 3 (-s 3), and baseline Zero (-bi Zero).
- Follow Resampling_Paper.ipynb to regenerate the plots in the paper.

Reach out to raphaelkim@fas.harvard.edu for any questions or additional setup files referenced in the scripts above. Per the IRB and consent agreements for HeartSteps V2/V3, the team is unable to share the HeartSteps V2/V3 data.
