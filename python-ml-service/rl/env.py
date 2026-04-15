"""
3주차: Gymnasium 커스텀 환경
위성이 제한된 스텝 내에서 변화가 큰 영역을 먼저 탐색하도록 학습
→ 서울다이나믹스 어필: PPO, 보상 설계, 시뮬레이션 환경
"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SatelliteScanEnv(gym.Env):
    """
    위성 관측 스케줄링 환경

    상태(State): 8x8 그리드 - 각 칸의 변화율 + 방문 여부
    행동(Action): 상하좌우 이동 (4방향)
    보상(Reward):
      - 변화율 높은 칸 방문: +변화율 값 (0~1)
      - 이미 방문한 칸 재방문: -0.5
      - 매 스텝: -0.01 (시간 비용)
      - 모든 변화 영역 발견: +2.0 보너스
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=8, max_steps=50, change_map=None):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps

        # 행동 공간: 0=위, 1=아래, 2=왼쪽, 3=오른쪽
        self.action_space = spaces.Discrete(4)

        # 관측 공간: [변화율 맵(8x8) + 방문 맵(8x8) + 에이전트 위치(8x8)]
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(3, grid_size, grid_size),
            dtype=np.float32
        )

        self.change_map = change_map  # 외부에서 주입 가능

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # 변화 맵 생성 (외부 주입 또는 랜덤)
        if self.change_map is not None:
            self.grid = self.change_map.copy()
        else:
            self.grid = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            # 랜덤 변화 영역 3~6개 생성
            num_changes = self.np_random.integers(3, 7)
            for _ in range(num_changes):
                x = self.np_random.integers(0, self.grid_size)
                y = self.np_random.integers(0, self.grid_size)
                value = self.np_random.uniform(0.3, 1.0)
                self.grid[y, x] = value

        # 방문 맵 초기화
        self.visited = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # 에이전트 시작 위치 (중앙)
        self.agent_pos = [self.grid_size // 2, self.grid_size // 2]
        self.visited[self.agent_pos[0], self.agent_pos[1]] = 1.0

        self.steps = 0
        self.total_reward = 0.0
        self.changes_found = 0
        self.total_changes = int(np.sum(self.grid > 0))

        return self._get_obs(), {}

    def _get_obs(self):
        agent_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        agent_map[self.agent_pos[0], self.agent_pos[1]] = 1.0
        return np.stack([self.grid, self.visited, agent_map], axis=0)

    def step(self, action):
        self.steps += 1

        # 이동 (경계 체크)
        dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
        new_y = np.clip(self.agent_pos[0] + dy, 0, self.grid_size - 1)
        new_x = np.clip(self.agent_pos[1] + dx, 0, self.grid_size - 1)
        self.agent_pos = [new_y, new_x]

        # 보상 계산
        reward = -0.01  # 시간 비용

        if self.visited[new_y, new_x] > 0:
            reward -= 0.5  # 재방문 패널티
        else:
            self.visited[new_y, new_x] = 1.0
            change_value = self.grid[new_y, new_x]
            if change_value > 0:
                reward += change_value  # 변화 영역 발견 보상
                self.changes_found += 1

        # 모든 변화 영역 발견 보너스
        if self.total_changes > 0 and self.changes_found >= self.total_changes:
            reward += 2.0

        self.total_reward += reward

        # 종료 조건
        terminated = self.changes_found >= self.total_changes and self.total_changes > 0
        truncated = self.steps >= self.max_steps

        info = {
            "changes_found": self.changes_found,
            "total_changes": self.total_changes,
            "steps": self.steps,
        }

        return self._get_obs(), reward, terminated, truncated, info


# Gymnasium 등록
gym.register(
    id="SatelliteScan-v0",
    entry_point="rl.env:SatelliteScanEnv",
    max_episode_steps=50,
)

if __name__ == "__main__":
    env = SatelliteScanEnv()
    obs, info = env.reset()
    print(f"관측 shape: {obs.shape}")
    print(f"변화 영역 수: {info.get('total_changes', 'N/A')}")

    # 랜덤 에이전트 테스트
    total = 0
    for _ in range(10):
        obs, info = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        total += info["changes_found"]
        print(f"발견: {info['changes_found']}/{info['total_changes']}, "
              f"스텝: {info['steps']}")

    print(f"\n랜덤 에이전트 평균 발견율: {total / 10:.1f}")
