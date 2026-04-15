/**
 * 분석 모듈: Controller + Service + Entity + Queue 통합
 */
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { BullModule } from '@nestjs/bull';
import { AnalysisController } from './analysis.controller';
import { AnalysisService } from './analysis.service';
import { AnalysisResult } from './analysis.entity';
import { AnalysisProcessor } from '../queue/queue.processor';

@Module({
  imports: [
    TypeOrmModule.forFeature([AnalysisResult]),
    BullModule.registerQueue({ name: 'analysis' }),
  ],
  controllers: [AnalysisController],
  providers: [AnalysisService, AnalysisProcessor],
  exports: [AnalysisService],
})
export class AnalysisModule {}
