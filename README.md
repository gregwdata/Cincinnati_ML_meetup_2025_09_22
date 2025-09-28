# Data Science in Motion

Some ML and non-ML Techniques for IoT Time-Series Analysis

Greg Wilson <a href="https://github.com/gregwdata" class="ns-c-iconlink"><mdi-github-circle/></a> <a href="https://www.linkedin.com/in/greg-wilson-6212572/" class="ns-c-iconlink"><mdi-linkedin-box/></a>  

This repo includes demo code that accompany the presentation "Data Science in Motion", given at the [Cincinnati Machine Learning Meetup](https://www.meetup.com/cincinnati-machine-learning-meetup/events/304736483/), 22 September, 2025.

The presentation can be viewed [here](https://gregwdata.github.io/Slides_Cincinnati_ML_meetup_2025_09_22)

## Contents

The files whose names start with the number 01 - 11 demonstrate various concepts of the presentation.

## Running yourself

The data streaming functions are intended to read data from a `phyphox` app with its REST API functionality enabled. The phone should be on the same wifi network as the machine you run this code from. Or you may find success directly hotlinking a connection betweeon your machine and phone.

Most of the scripts expect a `.env` file with a single variable:

```
PHYPHOX_ADDRESS=<ip address or hostname of your phone>
```

You can also directly set `PHYPHOX_ADDRESS` as an environment variable.

### Python environment

This project was run in an environment congfigured with `uv`. Use `uv sync` to creata local `.venv` folder with the required dependencies. 
