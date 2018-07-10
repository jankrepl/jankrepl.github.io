---
title: MCMC for generation animations
layout: single
author_profile: true
read_time: true
comments: true
share: null
related: true
excerpt: Sample the sample
classes: wide
tags:
- MCMC
- random
header:
  teaser: "/assets/images/MCMC_animation/teaser.gif"
---


![test]({{ "/assets/images/MCMC_animation/teaser.gif" | absolute_url }})


# Introduction
Markov Chain Monte Carlo (MCMC) is a widely popular technique in Bayesian statistics. It is used for posteriori distribution sampling since the analytical form is very often untrackable. In this post, however, we are going to use it to generate animations from static images/logos. Incidentally, it might serve as an introduction to MCMC and rejection sampling. The idea is based on a great open source package **imcmc** ([link](https://github.com/ColCarroll/imcmc)) that is built upon **PyMC3** ([link](https://github.com/pymc-devs/pymc3)).


# Preliminaries

While MCMC is applicable in any dimension, we are going to focus on 2D distributions. Why? Well, check the bellow diagram.

![medium_mc]({{ "/assets/images/MCMC_animation/img_to_pdf.png" | absolute_url }})

We start with an RGB image (preferably a simple logo), turn it into grayscale and assign each pixel a probability based on intensity. In our case, the darker the pixel the higher the probability. This results in a **discrete 2D distribution**.

Once we have this distribution we can draw samples (pixels) from it. In general, we look for the following 2 properties
1. The samples really come from the target distribution 
2. The succession of samples and their visualization is aesthetically pleasing

While the 2nd property is always in the eye of the beholder, the 1st one can be guaranteed by using appropriate sampling algorithms. In what follows, 3 sampling schemes are going to be described that can be readily applied in our setting.



# Sampling schemes
### Rejection sampling
The first method falls into the class of IID (**independent** and identically distributed) sampling methods. This esentially means that current sample will have no effect on what the next sample is going to be. The algorithm assumes we are able to sample from a so called proposal distribution. Many different proposal distributions can be used but the most common ones are uniform (finite support) and normal (infinite support). The algorithm reads:

<br>
<center><img src="/assets/images/MCMC_animation/rs_algo.png" alt="rejection" width="500" height="600"></center>
<br>

To minimize the probability of rejecting samples, it is essential that we select a proposal distribution that is as similar to our target as possible. Also, we want the scaling constant M to be as low as possible. See below (1D) scheme of rejection sampling.



![medium_mc]({{ "/assets/images/MCMC_animation/rejection_sampling.png" | absolute_url }})


In our setting, we can take a 2D uniform distribution to serve as the proposal.

```python
def rejection_sampling(image, approx_samples):

    image_pdf = image / image.sum()
    pdf_max = image_pdf.max()
    height, width = image_pdf.shape
    p_success = 1 / (height * width * pdf_max)
    actual_samples = min(int(approx_samples / p_success), int(1e8))

    samples_height = np.random.randint(0, high=height, size=actual_samples)
    samples_width = np.random.randint(0, high=width, size=actual_samples)
    samples_uniform = np.random.uniform(0, 1, size=actual_samples)

    result = [(h, w) for (h, w, u) in zip(samples_height, samples_width, samples_uniform) if
                    (image_pdf[h, w] >= pdf_max * u)]

    return result
```

In lower dimensions, rejection sampling offers a really good tradeoff between simplicity and performance. However, with increasing number of dimensions it suffers from the infamous **curse of dimensionality**. Luckily for us, 2D is not cursed and therefore rejection sampling is a great choice.


### Gibbs sampling
Gibbs sampling falls into the second category of samplers that generates samples via construction of a Markov chain. As a result, these samples are **not** independent. In fact, they are not even identically distribution until the chain reaches its stationary distribution. For that reason, it is a common practise to discard the first x samples to make sure that the chain "forgot" the starting point.

<br>
<center><img src="/assets/images/MCMC_animation/gibbs_algo.png" alt="gibbs" width="500" height="600"></center>
<br>

```python
def gibbs_sampling_single(image, w_start, samples):

    image_pdf = image / image.sum()
    height, width = image_pdf.shape
    result = []
    w_current = w_start

    for _ in range(samples):
        # sample height
        h_given_w = image_pdf[:, w_current] / image_pdf[:, w_current].sum()
        h_current = np.random.choice(np.array(range(height)), size=1, p=h_given_w)[0]

        # sample width
        w_given_h = image_pdf[h_current, :] / image_pdf[h_current, :].sum()
        w_current = np.random.choice(np.array(range(width)), size=1, p=w_given_h)[0]

        result.append((h_current, w_current))

    return result
```

### Metropolis-Hastings sampling
Similarly to Gibbs, Metropolis-Hastings sampling also creates a Markov chain. However, it is more general (Gibbs a special case) and flexible.

<br>
<center><img src="/assets/images/MCMC_animation/metropolis_algo.png" alt="metropolis" width="500" height="600"></center>
<br>

The intuition is the following. Given a current sample, the proposal distribution gives us a suggestion for a new sample. We then assess the eligibility of it by inspecting how much more (less) likely it is than the current sample and also take into account possible bias the proposal distribution might have towards this sample. Everything combined, we compute a probability of acception the new sample $r$ and then we let the randomness make the decision.

When it comes to the choice of the proposal distribution, there are a couple of very standard options
Choosing a symmetric proposal $\textbf{x}' \sim q(.|\textbf{x})=N(\textbf{x}, \Sigma)$(Normal centered at current state), 


# Animations

Let's finally look at some results, shall we? For each logo, rejection and Metropolis-Hastings sampling The MCMC sampling + visualization (creating gifs) was performed using the great package imcmc that builds upon pymc3. It uses the Metroplolis algorithm with a default proposal. The most important hyperparameters

* burn-in: 500
* samples: 10000


![medium_mc]({{ "/assets/images/MCMC_animation/organization.png" | absolute_url }})

![medium_mc]({{ "/assets/images/MCMC_animation/medium_final_s.gif" | absolute_url }})

![medium_mc]({{ "/assets/images/MCMC_animation/tm_final_s.gif" | absolute_url }})

![medium_mc]({{ "/assets/images/MCMC_animation/cisco_final_s.gif" | absolute_url }})

![medium_mc]({{ "/assets/images/MCMC_animation/git_final_s.gif" | absolute_url }})

![medium_mc]({{ "/assets/images/MCMC_animation/opencv_final_s.gif" | absolute_url }})

![medium_mc]({{ "/assets/images/MCMC_animation/scipy_final_s.gif" | absolute_url }})

![medium_mc]({{ "/assets/images/MCMC_animation/custom_final_s.gif" | absolute_url }})

### Comments

1. Each color represents a different markov chain. The color clusters within the animation demonstrate that successive samples are dependend and that it is necessary to use mutliple chains to sample from the entire distribution.

2. Samples generated by rejection sampling are independent.

3. For multimodal distributions Metropolis-Hastings might get stuck in certain regions. This is a very common problem and can be combatted via changing the proposal or sampling more/longer chains. Naturally, we can also venture into the wild and use some other samplers.


If you want to dig deeper and see the source code check this [**notebook**](https://github.com/jankrepl/creating-animations-with-MCMC/blob/master/main.ipynb). 


# References
1. imcmc ([https://github.com/ColCarroll/imcmc](https://github.com/ColCarroll/imcmc))
2. PyMC3 ([https://github.com/pymc-devs/pymc3](https://github.com/pymc-devs/pymc3))
3. Kevin P. Murphy. 2012. Machine Learning: A Probabilistic Perspective. The MIT Press.

  <script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
  </script>
  <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

