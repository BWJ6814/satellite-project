/**
 * 큐 프로세서: Bull에서 작업을 꺼내 Python ML 서비스 호출
 * → 루미르 어필: 메시지 큐 기반 비동기 처리, 대용량 데이터 파이프라인
 *
 * 흐름:
 * 1. 큐에서 작업 수신 (analysisId, 이미지 경로)
 * 2. Python ML에 변화 탐지 요청
 * 3. Python RL에 우선순위 요청
 * 4. 결과를 PostgreSQL에 저장
 */
import { Processor, Process } from '@nestjs/bull';
import { Job } from 'bull';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { AnalysisResult } from '../analysis/analysis.entity';
import axios from 'axios';
import * as fs from 'fs';
import * as FormData from 'form-data';

const ML_SERVICE_URL = 'http://localhost:8001';

@Processor('analysis')
export class AnalysisProcessor {
  constructor(
    @InjectRepository(AnalysisResult)
    private readonly repo: Repository<AnalysisResult>,
  ) {}

  @Process('process-images')
  async handleAnalysis(job: Job<{
    analysisId: string;
    beforePath: string;
    afterPath: string;
  }>) {
    const { analysisId, beforePath, afterPath } = job.data;

    console.log(`[큐] 분석 시작: ${analysisId}`);

    try {
      // 상태 업데이트: processing
      await this.repo.update(analysisId, { status: 'processing' });

      // ── 1단계: 변화 탐지 (Python ML 서비스 호출) ──
      console.log(`[큐] 변화 탐지 중...`);
      const formData = new FormData();
      formData.append('before', fs.createReadStream(beforePath));
      formData.append('after', fs.createReadStream(afterPath));

      const detectResponse = await axios.post(
        `${ML_SERVICE_URL}/api/detect-changes`,
        formData,
        {
          headers: formData.getHeaders(),
          timeout: 60000, // 60초 타임아웃
        },
      );
      const detectionResult = detectResponse.data;

      // ── 2단계: 탐색 우선순위 (Python RL 서비스 호출) ──
      console.log(`[큐] 우선순위 계산 중...`);
      let priorityResult = { priorities: [] };
      if (detectionResult.changes && detectionResult.changes.length > 0) {
        const prioResponse = await axios.post(
          `${ML_SERVICE_URL}/api/prioritize`,
          detectionResult.changes,
          { timeout: 30000 },
        );
        priorityResult = prioResponse.data;
      }

      // ── 3단계: 결과 저장 (PostgreSQL) ──
      await this.repo.update(analysisId, {
        status: 'completed',
        totalChangeRate: detectionResult.total_change_rate,
        numRegions: detectionResult.num_regions,
        changes: detectionResult.changes,
        priorities: priorityResult.priorities,
      });

      console.log(`[큐] 분석 완료: ${analysisId} (변화율: ${detectionResult.total_change_rate}%)`);

    } catch (error) {
      console.error(`[큐] 분석 실패: ${analysisId}`, error.message);

      await this.repo.update(analysisId, {
        status: 'failed',
        report: `분석 실패: ${error.message}`,
      });
    }
  }
}
