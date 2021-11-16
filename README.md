<h1> Can Intelligent Smart Homes change Human Behavior? </h1>
<a href="https://ieeexplore.ieee.org/document/9612040">Click here to read the paper </a>
<h2> Abstract </h2>
Smart homes are becoming increasingly popular as a result of advances in machine learning and cloud computing. Devices such as smart thermostats and speakers are now capable of learning from user feedback and adaptively adjust their settings to human preferences. Nonetheless, these devices might in turn impact human behavior. To investigate the potential impacts of smart homes on human behavior we simulate a series of Hierarchical-Reinforcement Learning-based human models capable of performing various activities namely setting temperature and humidity for thermal comfort inside a Q-Learning-based smart home model. We then investigate the possibility of the human models' behaviors being altered as a result of the smart home and the human model adapting to one another. For our human model, the activities are based on Hierarchical-Reinforcement Learning. This allows the human to learn how long it must continue a given activity and decide when to leave it. We then integrate our human model in the environment along with the smart home model and perform rigorous experiments considering various scenarios involving a model of a single human and models of two different humans with the smart home. Our experiments show that with the smart home, the human model can exhibit unexpected behaviors like frequent changing of activities and an increase in the time required to modify the thermal preferences. With two human models, we interestingly observe that certain combinations of models result in normal behaviours, while other combinations exhibit the same unexpected behaviors as those observed from the single human experiment.

<h2> Main Contribution </h2>
<ol>
  <li>
    We model a smart home with RL to learn to optimize ambient parameters, namely temperature and humidity for maximizing human agents' comfort. We then model a simulated human agent capable of pursuing and switching between various activities.
  </li>
 <li>
   We show that in such environments, the co-existence of the RL-based SHS and the human model can lead to unexpected changes in the optimal policy of the human model.
  </li>
  <li>
    We discover that unintended behaviors are present with models that have less overlapping comfort ranges and different reward structures.
  </li>
  </ol>
<p align="center">
  <img src="https://github.com/Shashi18/SmartHome-Changing-Human-Behavior/blob/master/Plots/HRLX-1.jpg" width=50%>
</p>
<h2>RL Smart Home with Single Human</h2>
<ol>
    <li> Our smart home model can successfully learn to anticipate thermal preference of a given model H_A, which does not show an anomaly in the presence of smart home. When integrated with the SHS, H_A spends fewer time-steps to change the TH without any unintended behaviour like excessive switching between activities.% or increase in the time-steps needed to change TH.
  </li>
  <li>
    When a human model with a slightly different thermal preference H_B is introduced in the environment, the SHS can learn and adapt according to its preference, thus also managing to reduce the number of time-steps needed to changed the TH.
  </li>
  <li>
    When a model with a different reward function and different TH preference H_C is integrated into the environment, it exhibits an unexpected behavior by frequently switching between activities, resulting in an increase in time-steps.
  </li>
  <li>
    When a model with a different reward function but similar TH preference H_D compared to the baseline model is integrated into the SHS, frequent switching between activities is also observed, and the time-steps to set the TH are relatively high. 
  </li>
<p align="center">
  <img src="https://github.com/Shashi18/SmartHome-Changing-Human-Behavior/blob/master/Plots/Exp1_2_3-1.jpg" width=50%>
  </p>
  <h6> Sample plots of activities over time for each model HA, HB, HC, and HD with and without the SHS. In (a), (c), (e), each model learns to complete the tasks without interruption. In (b) and (f), the SHS anticipates human preferences, speeding up the time for human models HA, HB, HD respectively to complete the activities. In (d), Model HC (different internal reward structure) behaves erratically in the presence of SHS. `<b>Set</b> <i>i</i>' denotes the action of setting TH for the corresponding activity <i>i</i>
  </h6>
  <p align="center">
  <img src="https://github.com/Shashi18/SmartHome-Changing-Human-Behavior/blob/master/Plots/Bar_123 (3)-1.jpg" width=30%>
   
    
  </p>
  <h6>
    Time-steps of human Models HA, HB, HC, and HD while setting TH for comfort, with and without the SHS.
  </h6>
  <p align="center">
   <img src="https://user-images.githubusercontent.com/15111631/141936668-b4fcb2f6-5628-4ebf-828d-563fdfca7f8f.png" width=65%>
  </p>
  <h6>
    Hierarchical 3 Part Value function for our human model.
  </h6>
   <img src="https://github.com/Shashi18/SmartHome-Changing-Human-Behavior/blob/master/Plots/Exp_1_2_3-1.jpg" width=100%>
  
  <h2> Conclusion </h2>
  For future work, this study can be used as a framework to be expanded and include real \textit{human-home interaction} data. Such an experiment would require monitoring of user behaviour over a period of time in the presence of a smart thermostat within a smart home. The study would also require a control group where no smart home systems are utilized. Data such as those recorded by activity trackers as well as thermal settings, collected from both groups, would then be used to improve the existing human models or even generate new ones for accurate representation of human behavior in the context of decision making as influenced by smart home systems. With an improved RL-based human model, the arbitrariness of the human behaviour especially when interacting with smart home systems through a longitudinal study could be better understood. Lastly, our study shows that human adaptability could be an important factor to take into account when designing a smart home system. Developing a reliable and effective  model of human adaptability could be critical to best predict the smart home performance in real contexts and it's potential implications and impacts on human behaviours.

