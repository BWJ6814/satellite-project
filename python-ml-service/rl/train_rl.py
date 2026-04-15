"""
3주차: PPO 에이전트 학습
Stable-Baselines3로 위성 탐색 최적화 에이전트 훈련
CPU에서 약 10~20분 소요
→ 서울다이나믹스 어필: PPO, SAC, 보상 설계, Stable-Baselines3
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from rl.env import SatelliteScanEnv

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "saved_models")


def train_ppo():
    os.makedirs(MODEL_DIR, exist_ok=True)

    print("=" * 50)
    print("PPO 에이전트 학습 시작")
    print("=" * 50)

    # 벡터화된 환경 생성 (병렬 학습)
    env = make_vec_env(SatelliteScanEnv, n_envs=4)

    # 평가용 환경
    eval_env = make_vec_env(SatelliteScanEnv, n_envs=1)

    # PPO 모델 생성
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,           # 할인율
        gae_lambda=0.95,      # GAE 파라미터
        clip_range=0.2,       # PPO 클리핑 범위
        ent_coef=0.01,        # 엔트로피 계수 (탐험 장려)
        verbose=1,
    )

    # 콜백: 주기적으로 평가하고 최고 모델 저장
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=MODEL_DIR,
        eval_freq=2000,
        n_eval_episodes=10,
        deterministic=True,
    )

    # 학습 (총 50,000 스텝)
    print("\n학습 시작... (CPU에서 약 10~20분)")
    model.learn(
        total_timesteps=50000,
        callback=eval_callback,
    )

    # 최종 모델 저장
    model.save(os.path.join(MODEL_DIR, "ppo_satellite"))
    print(f"\n학습 완료! 모델 저장: {MODEL_DIR}/ppo_satellite.zip")

    # 학습된 에이전트 vs 랜덤 에이전트 비교
    print("\n" + "=" * 50)
    print("성능 비교: PPO vs 랜덤")
    print("=" * 50)

    test_env = SatelliteScanEnv()

    # PPO 에이전트 테스트
    ppo_found = []
    ppo_steps = []
    for _ in range(20):
        obs, info = test_env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
        ppo_found.append(info["changes_found"])
        ppo_steps.append(info["steps"])

    # 랜덤 에이전트 테스트
    random_found = []
    random_steps = []
    for _ in range(20):
        obs, info = test_env.reset()
        done = False
        while not done:
            action = test_env.action_space.sample()
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
        random_found.append(info["changes_found"])
        random_steps.append(info["steps"])

    import numpy as np
    print(f"PPO   평균 발견: {np.mean(ppo_found):.1f}, 평균 스텝: {np.mean(ppo_steps):.1f}")
    print(f"랜덤  평균 발견: {np.mean(random_found):.1f}, 평균 스텝: {np.mean(random_steps):.1f}")
    improvement = (np.mean(ppo_found) - np.mean(random_found)) / max(np.mean(random_found), 1) * 100
    print(f"→ PPO가 랜덤 대비 {improvement:.0f}% 더 많이 발견")


if __name__ == "__main__":
    train_ppo()
