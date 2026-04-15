/**
 * Nest.js 백엔드 서버 진입점
 * → 루미르 어필: Nest.js + TypeScript 기반 백엔드
 */
import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';
import { ValidationPipe } from '@nestjs/common';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);

  // CORS 설정
  app.enableCors();

  // 유효성 검증 파이프
  app.useGlobalPipes(new ValidationPipe({ transform: true }));

  // Swagger API 문서
  const config = new DocumentBuilder()
    .setTitle('위성 이미지 분석 플랫폼 API')
    .setDescription('SAR 위성 이미지 변화 탐지 및 분석 리포트 생성')
    .setVersion('1.0')
    .build();
  const document = SwaggerModule.createDocument(app, config);
  SwaggerModule.setup('api', app, document);

  await app.listen(3000);
  console.log('Nest.js 서버 실행: http://localhost:3000');
  console.log('API 문서: http://localhost:3000/api');
}
bootstrap();
