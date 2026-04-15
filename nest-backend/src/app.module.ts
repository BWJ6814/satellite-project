/**
 * 앱 루트 모듈
 * → 루미르 어필: TypeORM, PostgreSQL, Redis(Bull) 메시지 큐
 */
import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { BullModule } from '@nestjs/bull';
import { AnalysisModule } from './analysis/analysis.module';

@Module({
  imports: [
    // PostgreSQL 연결 (루미르 기술스택: PostgreSQL + TypeORM)
    TypeOrmModule.forRoot({
      type: 'postgres',
      host: 'localhost',
      port: 5432,
      username: 'admin',
      password: 'admin1234',
      database: 'satellite_db',
      autoLoadEntities: true,
      synchronize: true, // 개발용. 프로덕션에서는 false
    }),

    // Redis + Bull 메시지 큐 (루미르 기술스택: RabbitMQ/Kafka 대응)
    BullModule.forRoot({
      redis: {
        host: 'localhost',
        port: 6379,
      },
    }),

    // 분석 모듈
    AnalysisModule,
  ],
})
export class AppModule {}
