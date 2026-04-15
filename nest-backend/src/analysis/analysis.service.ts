/**
 * 분석 서비스: 이미지 업로드 → 큐 등록 → Python ML 호출 → 결과 저장
 * → 루미르 어필: 대용량 데이터 파이프라인, 비동기 처리, API 설계
 */
import { Injectable } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { InjectQueue } from '@nestjs/bull';
import { Queue } from 'bull';
import { AnalysisResult } from './analysis.entity';
import axios from 'axios';
import * as fs from 'fs';
import * as path from 'path';
import * as FormData from 'form-data';

const ML_SERVICE_URL = 'http://localhost:8001';
const UPLOAD_DIR = path.join(__dirname, '..', '..', 'uploads');

@Injectable()
export class AnalysisService {
  constructor(
    @InjectRepository(AnalysisResult)
    private readonly repo: Repository<AnalysisResult>,
    @InjectQueue('analysis')
    private readonly analysisQueue: Queue,
  ) {
    // 업로드 폴더 생성
    if (!fs.existsSync(UPLOAD_DIR)) {
      fs.mkdirSync(UPLOAD_DIR, { recursive: true });
    }
  }

  /**
   * 1. 이미지 업로드 + 분석 작업을 큐에 등록
   * → 루미르: "위성에서 전송되는 대용량 데이터 처리 파이프라인 구축"
   */
  async createAnalysis(
    beforeFile: Express.Multer.File,
    afterFile: Express.Multer.File,
    region?: string,
  ): Promise<AnalysisResult> {
    // 파일 저장
    const beforePath = path.join(UPLOAD_DIR, `before_${Date.now()}.png`);
    const afterPath = path.join(UPLOAD_DIR, `after_${Date.now()}.png`);
    fs.writeFileSync(beforePath, beforeFile.buffer);
    fs.writeFileSync(afterPath, afterFile.buffer);

    // DB에 레코드 생성 (상태: pending)
    const analysis = this.repo.create({
      region: region || '미지정',
      status: 'pending',
      beforeImagePath: beforePath,
      afterImagePath: afterPath,
    });
    const saved = await this.repo.save(analysis);

    // Bull 큐에 작업 등록 (비동기 처리)
    await this.analysisQueue.add('process-images', {
      analysisId: saved.id,
      beforePath,
      afterPath,
    });

    return saved;
  }

  /**
   * 2. Python ML 서비스 호출 → 변화 탐지
   */
  async runChangeDetection(beforePath: string, afterPath: string): Promise<any> {
    const formData = new FormData();
    formData.append('before', fs.createReadStream(beforePath));
    formData.append('after', fs.createReadStream(afterPath));

    const response = await axios.post(
      `${ML_SERVICE_URL}/api/detect-changes`,
      formData,
      { headers: formData.getHeaders() },
    );
    return response.data;
  }

  /**
   * 3. Python RL 서비스 호출 → 탐색 우선순위
   */
  async runPrioritization(changes: any[]): Promise<any> {
    const response = await axios.post(
      `${ML_SERVICE_URL}/api/prioritize`,
      changes,
    );
    return response.data;
  }

  /**
   * 4. Python LLM 서비스 호출 → 리포트 생성
   */
  async generateReport(analysisResults: any[], question: string): Promise<string> {
    const response = await axios.post(
      `${ML_SERVICE_URL}/api/generate-report`,
      { analysis_results: analysisResults, question },
    );
    return response.data.report;
  }

  /**
   * 5. 분석 결과 업데이트
   */
  async updateResult(id: string, data: Partial<AnalysisResult>): Promise<void> {
    await this.repo.update(id, data);
  }

  /**
   * 결과 조회 API들
   * → 루미르: "SAR 영상분석 플랫폼 백엔드 API 설계 및 개발"
   */
  async findAll(): Promise<AnalysisResult[]> {
    return this.repo.find({ order: { createdAt: 'DESC' } });
  }

  async findOne(id: string): Promise<AnalysisResult> {
    return this.repo.findOneBy({ id });
  }

  async findByRegion(region: string): Promise<AnalysisResult[]> {
    return this.repo.find({
      where: { region },
      order: { createdAt: 'DESC' },
    });
  }
}
