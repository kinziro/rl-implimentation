## 使用する記号

- a：行動
- s：状態
- z：埋め込み空間の値
- t：タスクID (ワンホット表現)

## 使用するネットワーク
以下の3つのネットワークを学習に使用

- ポリシーネットワーク：$\pi_\theta(a|s, z)$
- 埋め込みネットワーク：$p_\phi (z|t)$
- 推論ネットワーク：$q_\psi (z|a, s^H)$

## Loss関数

```math
\begin{aligned}
L(\theta,\phi,\psi)=E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[ \sum_{i=0}^\infty \gamma^i \hat{r}(s_i,a_i,z,t)|s_{i+1} \sim p(s_{i+1}|a_i,s_i) \Biggr]+\alpha_1 E_{t \in T} \Biggr[H[p_{\phi} (z|t)] \Biggr]
\\\\   

where \quad \hat{r}(s_i,a_i,z,t)=\biggr[r_t(s_i,a_i)+ \alpha_2 \log q_\psi(z|a_i,s_i^H) + \alpha_3 H[\pi_\theta(a|s_i,z)] \biggr]

\end{aligned}
```

## パラメータごとのLossの微分

### $\theta$について微分

```math
R(\theta) = \sum_{i=0}^\infty \gamma^i \hat{r}(s_i,a_i,z,t)
```

と置くと

```math
\begin{aligned}
\nabla_\theta L(\theta) &= \nabla_\theta E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\theta) \Biggr] \\
& = \nabla_\theta \iiint P(t) \prod_i^\infty \pi_{\theta}(a_i,z|s_i,t) R(\theta) da dz dt \\
&= \iiint P(t) \nabla_\theta \prod_i^\infty \pi_{\theta}(a_i,z|s_i,t)R(\theta) da dz dt \\
&= \iiint P(t) \nabla_\theta \prod_i^\infty \pi_{\theta}(a_i|z,s_i,t)p_\phi(z|t)R(\theta) da dz dt \\\\
&= \iiint P(t)p_\phi(z|t) \nabla_\theta \prod_i^\infty \pi_{\theta}(a_i|z,s_i,t)R(\theta) da dz dt \\\\
ここで \quad \nabla_x f(x) &= f(x) \nabla_x \log f(x) \quad より \\\\

&= \iiint P(t)p_\phi(z|t) \prod_i^\infty \pi_{\theta}(a_i|z,s_i,t) R(\theta) \nabla_\theta \log \prod_i^\infty \pi_{\theta}(a_i|z,s_i,t)R(\theta) da dz dt \\
&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\theta) \nabla_\theta \log \prod_i^\infty \pi_{\theta}(a_i|z,s_i,t)R(\theta) \Biggr] \\
&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\theta) \biggr(  \nabla_\theta \sum_i^\infty \log \pi_{\theta}(a_i|z,s_i,t) +  \nabla_\theta \log R(\theta) \biggr) \Biggr] \\
&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\theta)  \nabla_\theta \sum_i^\infty \log \pi_{\theta}(a_i|z,s_i,t) +  \nabla_\theta R(\theta) \Biggr] \\
&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\theta)  \nabla_\theta \sum_i^\infty \log \pi_{\theta}(a_i|z,s_i,t) +  \nabla_\theta \sum_i^\infty \gamma^i \biggr( \alpha_2 \log q_\psi(z|a_{i, \theta},s_i^H) + \alpha_3 H[\pi_\theta(a|s_i,z)] \biggr) \Biggr] \\

&\simeq E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\theta)  \nabla_\theta \sum_i^\infty \log \pi_{\theta}(a_i|s_i,z) + \nabla_\theta \sum_i^\infty \gamma^i \biggr( \alpha_2 \log q_\psi(z|a_{i, \theta},s_i^H) + \alpha_3 H[\pi_\theta(a|s_i,z)] \biggr) \Biggr] \\ 

\end{aligned}
```
分散環境によりK種類のタスクをD個ずつ並列してデータを収集する場合を考える。
この際、Iステップごとにlossの微分を計算する場合、サンプリングベースでは

```math
\begin{aligned}
\nabla_\theta L(\theta) &= \frac{1}{K} \frac{1}{D} \sum_k^K \sum_d^D \Biggr( R(\theta)  \nabla_\theta \sum_i^I \log \pi_{\theta}(a_i|s_i,z) + \nabla_\theta \sum_i^I \gamma^i \biggr( \alpha_2 \log q_\psi(z|a_{i, \theta},s_i^H) + \alpha_3 H[\pi_\theta(a|s_i,z)] \biggr) \Biggr)
\end{aligned}
```


### $\psi$について微分

```math
R(\psi) = \sum_{i=0}^\infty \gamma^i \hat{r}(s_i,a_i,z,t)
```

と置くと

```math
\begin{aligned}
\nabla_\psi L(\psi) &= \nabla_\psi E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\psi) \Biggr] \\
& = \nabla_\psi \iiint P(t) \prod_i^\infty \pi_{\theta}(a_i,z|s_i,t) R(\psi) da dz dt \\
&= \iiint p(t) \prod_i^\infty \pi_{\theta}(a_i,z|s_i,t) \nabla_\psi R(\psi) da dz dt \\
&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[\nabla_\psi R(\psi) \Biggr] \\
&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[\nabla_\psi \sum_i^\infty \gamma^i \alpha_2 \log q_\psi(z|a_i,s_i^H) \Biggr] \\


\end{aligned}
```
分散環境によりK種類のタスクをD個ずつ並列してデータを収集する場合を考える。
この際、Iステップごとにlossの微分を計算する場合、サンプリングベースでは

```math
\begin{aligned}
\nabla_\psi L(\theta) &= \frac{1}{K} \frac{1}{D} \sum_k^K \sum_d^D \Biggr( \nabla_\psi \sum_i^I \gamma^i \alpha_2 \log q_\psi(z|a_i,s_i^H) \Biggr)
\end{aligned}
```

### $\phi$について微分

```math
R(\phi) = \sum_{i=0}^\infty \gamma^i \hat{r}(s_i,a_i,z,t)
```

と置くと

```math
\begin{aligned}
\nabla_\phi L(\phi) &= \nabla_\phi E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\phi) \Biggr] + \alpha_1 \nabla_\phi E_{t \in T} \Biggr[ H \biggr[ p_\phi (z|t) \biggr] \Biggr] \\

& = \nabla_\phi \iiint P(t) \prod_i^\infty \pi_{\theta}(a_i|z_\phi,s_i,t)p_\phi(z|t) R(\phi) da dz dt + \alpha_1 \nabla_\phi \int P(t) H \biggr[ p_\phi (z|t) \biggr] \\

&= \iiint P(t) \nabla_\phi \prod_i^\infty \pi_{\theta}(a_i|z_\phi,s_i,t)p_\phi(z|t)R(\phi) da dz dt + \alpha_1 \int P(t) \nabla_\phi H \biggr[ p_\phi (z|t) \biggr] \\\\
ここで \quad \nabla_x f(x) &= f(x) \nabla_x \log f(x) \quad より \\\\

&= \iiint P(t) \prod_i^\infty \pi_{\theta}(a_i|z_\phi,s_i,t)p_\phi(z|t) R(\phi) \nabla_\phi \log \prod_i^\infty \pi_{\theta}(a_i|z_\phi,s_i,t)p_\phi(z|t)R(\phi) da dz dt + E_{t \in T} \Biggr[ \nabla_\phi H \biggr[ p_\phi (z|t) \biggr] \Biggr] \\

&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\phi) \nabla_\phi \log \prod_i^\infty \pi_{\theta}(a_i|z_\phi,s_i,t)p_\phi(z|t)R(\phi) \Biggr] + E_{t \in T} \Biggr[ \nabla_\phi H \biggr[ p_\phi (z|t) \biggr] \Biggr]\\

&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\phi) \biggr(  \nabla_\phi \sum_i^\infty \log \pi_{\theta}(a_i|z_\phi,s_i,t) + \nabla_\phi \log p_\phi(z|t) + \nabla_\phi \log R(\phi) \biggr) \Biggr] + E_{t \in T} \Biggr[ \nabla_\phi H \biggr[ p_\phi (z|t) \biggr] \Biggr] \\

&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\phi) \biggr(  \nabla_\phi \sum_i^\infty \log \pi_{\theta}(a_i|z_\phi,s_i,t) + \nabla_\phi \log p_\phi(z|t) \biggr) +  \nabla_\phi R(\phi) \Biggr] + E_{t \in T} \Biggr[ \nabla_\phi H \biggr[ p_\phi (z|t) \biggr] \Biggr] \\

&= E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\phi) \biggr(  \nabla_\phi \sum_i^\infty \log \pi_{\theta}(a_i|z_\phi,s_i,t) + \nabla_\phi \log p_\phi(z|t) \biggr) \\ 
&\qquad + \nabla_\phi \sum_i^\infty \gamma^i \biggr( \alpha_2 \log q_\psi(z|a_{i,\phi},s_i^H) + \alpha_3 H[\pi_\theta(a|s_i,z_\phi)] \biggr) \Biggr] + E_{t \in T} \Biggr[ \nabla_\phi H \biggr[ p_\phi (z|t) \biggr] \Biggr] \\

&\simeq E_{\pi_{\theta}(a,z|s,t), t \in T} \Biggr[R(\phi) \biggr(  \nabla_\phi \sum_i^\infty \log \pi_{\theta}(a_i|s_i,z_\phi) + \nabla_\phi \log p_\phi(z|t) \biggr) \\ 
&\qquad + \nabla_\phi \sum_i^\infty \gamma^i \biggr( \alpha_2 \log q_\psi(z|a_{i,\phi},s_i^H) + \alpha_3 H[\pi_\theta(a|s_i,z_\phi)] \biggr) \Biggr] + E_{t \in T} \Biggr[ \nabla_\phi H \biggr[ p_\phi (z|t) \biggr] \Biggr] \\



\end{aligned}
```
分散環境によりK種類のタスクをD個ずつ並列してデータを収集する場合を考える。
この際、Iステップごとにlossの微分を計算する場合、サンプリングベースでは

```math
\begin{aligned}
\nabla_\phi L(\phi) &= \frac{1}{K} \frac{1}{D} \sum_k^K \sum_d^D \Biggr( R(\phi) \biggr(  \nabla_\phi \sum_i^I \log \pi_{\theta}(a_i|s_i,z_\phi) + \nabla_\phi \log p_\phi(z|t) \biggr) \\ 
&\qquad + \nabla_\phi \sum_i^I \gamma^i \biggr( \alpha_2 \log q_\psi(z|a_{i,\phi},s_i^H) + \alpha_3 H[\pi_\theta(a|s_i,z_\phi)] \biggr) \Biggr) + \frac{1}{K} \sum_k^K \Biggr( \nabla_\phi H \biggr[ p_\phi (z|t) \biggr] \Biggr)

\end{aligned}
```


