/**
 * REST API 컨트롤러
 * → 루미르 어필: REST API 설계, Swagger 문서화
 */
import {
  Controller, Post, Get, Param, Query, Body,
  UseInterceptors, UploadedFiles,
} from '@nestjs/common';
import { FilesInterceptor } from '@nestjs/platform-express';
import { ApiTags, ApiOperation, ApiConsumes } from '@nestjs/swagger';
import { AnalysisService } from './analysis.service';

@ApiTags('분석')
@Controller('analysis')
export class AnalysisController {
  constructor(private readonly service: AnalysisService) {}

  @Post('upload')
  @ApiOperation({ summary: '위성 이미지 업로드 및 분석 요청' })
  @ApiConsumes('multipart/form-data')
  @UseInterceptors(FilesInterceptor('images', 2))
  async uploadAndAnalyze(
    @UploadedFiles() files: Express.Multer.File[],
    @Body('region') region?: string,
  ) {
    if (!files || files.length < 2) {
      return { error: 'before, after 이미지 2장이 필요합니다' };
    }
    const result = await this.service.createAnalysis(files[0], files[1], region);
    return {
      message: '분석 작업이 큐에 등록되었습니다',
      analysisId: result.id,
      status: result.status,
    };
  }

  @Get()
  @ApiOperation({ summary: '전체 분석 결과 목록 조회' })
  async findAll() {
    return this.service.findAll();
  }

  @Get(':id')
  @ApiOperation({ summary: '특정 분석 결과 조회' })
  async findOne(@Param('id') id: string) {
    return this.service.findOne(id);
  }

  @Get('region/:region')
  @ApiOperation({ summary: '지역별 분석 결과 조회' })
  async findByRegion(@Param('region') region: string) {
    return this.service.findByRegion(region);
  }

  @Post(':id/report')
  @ApiOperation({ summary: 'LLM 분석 리포트 생성 요청' })
  async generateReport(
    @Param('id') id: string,
    @Body('question') question: string,
  ) {
    const analysis = await this.service.findOne(id);
    if (!analysis) return { error: '분석 결과를 찾을 수 없습니다' };

    const report = await this.service.generateReport(
      analysis.changes || [],
      question || '전체 분석 결과를 요약해주세요',
    );

    await this.service.updateResult(id, { report });
    return { report };
  }
}
