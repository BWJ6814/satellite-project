/**
 * 분석 결과 DB 엔티티
 * → 루미르 어필: TypeORM 엔티티 설계, PostgreSQL 스키마
 */
import { Entity, PrimaryGeneratedColumn, Column, CreateDateColumn } from 'typeorm';

@Entity('analysis_results')
export class AnalysisResult {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'varchar', length: 255, nullable: true })
  region: string;

  @Column({ type: 'varchar', length: 50, default: 'pending' })
  status: string; // pending, processing, completed, failed

  @Column({ type: 'float', default: 0 })
  totalChangeRate: number;

  @Column({ type: 'int', default: 0 })
  numRegions: number;

  @Column({ type: 'jsonb', nullable: true })
  changes: any; // 변화 영역 상세 데이터

  @Column({ type: 'jsonb', nullable: true })
  priorities: any; // RL 우선순위 결과

  @Column({ type: 'text', nullable: true })
  report: string; // LLM 생성 리포트

  @Column({ type: 'varchar', length: 500, nullable: true })
  beforeImagePath: string;

  @Column({ type: 'varchar', length: 500, nullable: true })
  afterImagePath: string;

  @CreateDateColumn()
  createdAt: Date;
}
