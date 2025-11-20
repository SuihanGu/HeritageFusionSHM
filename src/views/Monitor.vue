<template>
  <div class="shm-monitor-container">
    <!-- Sensor Layout Diagram Card -->
    <div class="image-container">
      <div class="image-content-wrapper">
        <!-- Image -->
        <div class="image-wrapper">
          <img src="/image/yuhuang/image.png" alt="Sensor Layout Diagram" class="sensor-layout-image" />
        </div>
        
        <!-- Title and Refresh Information -->
        <div class="header-section">
          <div class="header">
            <h2>Chinese Heritage Building - Yuhuang Pavilion Structural Health Monitoring</h2>
            <div class="refresh-info">
              <span>Data Update Time: {{ lastUpdateTime }}</span>
              <el-button size="small" @click="refreshData">Manual Refresh</el-button>
            </div>
          </div>
          
          <!-- Building Introduction and History -->
          <div class="building-intro">
            <h3>Building Introduction</h3>
            <p>Yuhuang Pavilion, also known as Jingbian Tower, is located on the north city wall of Yu County, Hebei Province. It is a Ming Dynasty religious building dedicated to the Jade Emperor and was listed as a national key cultural relic protection unit in 1996. The building was first built in the tenth year of Ming Hongwu (1377), presided over by Zhou Fang, commander of Yuzhou Wei. It faces the east, west, and south gates from a distance, and had defensive functions in its early days. According to "Yuzhou Zhi", there were 24 towers on the city wall in the past, but this pavilion was the most magnificent and majestic.</p>
            
            <h3>Main Features</h3>
            <p>The building complex faces south, with a total area of about 2,000 square meters. It adopts a "three visible, two hidden" layout pattern. The main hall has a triple-eave Xieshan glazed tile roof, with a statue of the Jade Emperor and murals inside. There are existing Ming and Qing restoration steles and the Ming Dynasty "Tianxianzi" poem stele. Inside the pavilion, there is a bell tower on the left and a drum tower on the right, both are double-eave Xieshan tiled-roof square pavilion-style buildings. The brick and stone walls are decorated with two dragons playing with a pearl relief. Under the front eaves of the main hall, there are eight Ming and Qing steles, among which the Tianxianzi stele is engraved with poems by Su Zhigao during the Ming Jiajing period.
              The existing building retains the Ming Dynasty large timber structure characteristics. The hall is divided into upper, middle, and lower three-story pavilions, with a four-sided veranda in the middle floor. Iron bells hang from the four corners of the roof ridge, and the overall layout is symmetrical and rigorous.</p>
         </div>
        </div>
      </div>
    </div>

    <!-- Crack Meter - Full Width Row -->
    <div class="chart-container full-width">
      <div class="chart-title">Crack Meter</div>
      <div ref="crackChartRef" class="chart"></div>
    </div>

    <!-- Other Sensors - Two Column Layout -->
    <div class="chart-row">
      <!-- Inclinometer Probe - X Direction -->
      <div class="chart-container half-width">
        <div class="chart-title">Inclinometer Probe - X Direction</div>
        <div ref="tiltXChartRef" class="chart"></div>
      </div>

      <!-- Inclinometer Probe - Y Direction -->
      <div class="chart-container half-width">
        <div class="chart-title">Inclinometer Probe - Y Direction</div>
        <div ref="tiltYChartRef" class="chart"></div>
      </div>
    </div>

    <div class="chart-row">
      <!-- Level Meter -->
      <div class="chart-container half-width">
        <div class="chart-title">Level Meter</div>
        <div ref="levelChartRef" class="chart"></div>
      </div>

      <!-- Water Level Meter -->
      <div class="chart-container half-width">
        <div class="chart-title">Water Level Meter</div>
        <div ref="waterLevelChartRef" class="chart"></div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from "vue";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import axios from "axios";

// Register ECharts Components
echarts.use([
  LineChart,
  GridComponent,
  LegendComponent,
  TooltipComponent,
  DataZoomComponent,
  CanvasRenderer
]);

// Interface Definitions
interface CrackDataPoint {
  id?: number;
  time?: string;
  timestamp?: string | number;
  channel?: string;
  code?: string;
  number?: string;
  data1?: number | string;
  data2?: number | string;
  data3?: number | string;
}

interface TiltDataPoint {
  id?: number;
  time?: string;
  timestamp?: string | number;
  code?: string;
  number?: string;
  data1?: number | string; // X direction
  data2?: number | string; // Y direction
  data3?: number | string; // Temperature
}

interface LevelDataPoint {
  id?: number;
  time?: string;
  timestamp?: string | number;
  number?: string;
  data1?: number | string; // Settlement
  data2?: number | string;
  data3?: number | string | null;
}

interface WaterLevelDataPoint {
  id?: number;
  time?: string;
  timestamp?: string | number;
  number?: string;
  data1?: number | string; // Water level (mm)
  data2?: number | string;
  data3?: number | string;
}

// Device Numbers
const CRACK_NUMBERS = ["623622", "623628", "623641"];
const TILT_NUMBERS = ["00476464", "00476465", "00476466", "00476467"];
const LEVEL_NUMBERS = ["004521", "004548", "004591", "152947"];

// Chart References
const crackChartRef = ref<HTMLDivElement>();
const tiltXChartRef = ref<HTMLDivElement>();
const tiltYChartRef = ref<HTMLDivElement>();
const levelChartRef = ref<HTMLDivElement>();
const waterLevelChartRef = ref<HTMLDivElement>();

// Chart Instances
let crackChart: echarts.ECharts | null = null;
let tiltXChart: echarts.ECharts | null = null;
let tiltYChart: echarts.ECharts | null = null;
let levelChart: echarts.ECharts | null = null;
let waterLevelChart: echarts.ECharts | null = null;

// Data Refresh Related
const lastUpdateTime = ref<string>("");
let refreshTimer: number | null = null;
const REFRESH_INTERVAL = 10 * 60 * 1000; // 10 minutes

// ==========================================
// Unified Data Processing Utility Functions
// ==========================================

/**
 * Convert timestamp to milliseconds (ECharts requires millisecond timestamps)
 * @param timestamp - Unix second-level timestamp (may be string or number)
 * @returns millisecond timestamp
 */
function normalizeTimestamp(timestamp: string | number | undefined): number | null {
  if (timestamp === undefined || timestamp === null) {
    return null;
  }
  
  // If it's a string, convert to number first
  const ts = typeof timestamp === "string" ? parseFloat(timestamp) : timestamp;
  
  // Check if it's a valid number
  if (isNaN(ts) || ts <= 0) {
    return null;
  }
  
  // If timestamp is less than 10000000000, it's a second-level timestamp, need to multiply by 1000
  // If timestamp is greater than 10000000000, it's already a millisecond-level timestamp
  if (ts < 10000000000) {
    return ts * 1000;
  }
  
  return ts;
}

/**
 * Convert value to number (handle string-type values)
 * @param value - May be number or string
 * @returns number or null
 */
function normalizeValue(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined) {
    return null;
  }
  
  // If it's a string, try to convert to number
  if (typeof value === "string") {
    const num = parseFloat(value);
    return isNaN(num) ? null : num;
  }
  
  // If it's a number, check if it's NaN
  return isNaN(value) ? null : value;
}

// Get current timestamp (UNIX format, seconds)
function getCurrentTimestamp(): number {
  return Math.floor(Date.now() / 1000);
}

// Get timestamp range for past 24 hours
function getTimeRange(): { timestamp1: number; timestamp2: number } {
  const now = getCurrentTimestamp();
  const dayAgo = now - 24 * 60 * 60; // 24 hours ago
  return {
    timestamp1: dayAgo,
    timestamp2: now
  };
}

// API Call Functions
async function fetchCrackData() {
  const { timestamp1, timestamp2 } = getTimeRange();
  const response = await axios.get<CrackDataPoint[]>(
    "http://139.159.136.213:4999/iem/shm/jmData",
    {
      params: { timestamp1, timestamp2 },
      timeout: 15000
    }
  );
  return Array.isArray(response.data) ? response.data : [];
}

// 获取预测数据
interface PredictionPoint {
  time: string;
  timestamp: number;
  crack_1: number;
  crack_2: number;
  crack_3: number;
}

interface PredictionResponse {
  success: boolean;
  data?: {
    predictions: PredictionPoint[];
    timestamp: number;
  };
  message?: string;
}

async function fetchPredictionData() {
  try {
    const response = await axios.get<PredictionResponse>(
      "http://localhost:5000/api/predictions/crack",
      {
        timeout: 10000
      }
    );
    if (response.data.success && response.data.data) {
      return response.data.data.predictions;
    }
    return [];
  } catch (error) {
    console.warn("Failed to fetch prediction data:", error);
    return [];
  }
}

async function fetchTiltData() {
  const { timestamp1, timestamp2 } = getTimeRange();
  const response = await axios.get<TiltDataPoint[]>(
    "http://139.159.136.213:4999/iem/shm/jmBus",
    {
      params: { timestamp1, timestamp2 },
      timeout: 15000
    }
  );
  return Array.isArray(response.data) ? response.data : [];
}

async function fetchLevelData() {
  const { timestamp1, timestamp2 } = getTimeRange();
  const response = await axios.get<LevelDataPoint[]>(
    "http://139.159.136.213:4999/iem/shm/jmLevel",
    {
      params: { timestamp1, timestamp2 },
      timeout: 15000
    }
  );
  return Array.isArray(response.data) ? response.data : [];
}

async function fetchWaterLevelData() {
  const { timestamp1, timestamp2 } = getTimeRange();
  const response = await axios.get<WaterLevelDataPoint[]>(
    "http://139.159.136.213:4999/iem/shm/jmWlg",
    {
      params: { timestamp1, timestamp2 },
      timeout: 15000
    }
  );
  return Array.isArray(response.data) ? response.data : [];
}

// Create ECharts Configuration (Reference Stacked Line Chart Style)
function createChartOption(
  seriesData: Array<{ name: string; data: Array<[number, number | null]> }>,
  unit: string = "",
  predictionSeriesData?: Array<{ name: string; data: Array<[number, number | null]> }>
) {
  const colors = ["#5470C6", "#91CC75", "#FAC858", "#EE6666", "#73C0DE", "#3BA272", "#FC8452", "#9A60B4"];
  
  const series = seriesData.map((item, idx) => {
    const color = colors[idx % colors.length];
    return {
      name: item.name,
      type: "line",
      // Remove stack configuration, use independent line chart instead of stacked chart
      showSymbol: false,
      data: item.data,
      color: color,
      lineStyle: {
        width: 2,
        color: color
      },
      itemStyle: {
        color: color
      },
      smooth: 0.4,
      connectNulls: true
    };
  });

  // Add prediction data series (dashed line)
  if (predictionSeriesData && predictionSeriesData.length > 0) {
    const predictionSeries = predictionSeriesData
      .filter(item => item.data && item.data.length > 0) // Only add prediction series with data
      .map((item, idx) => {
        const color = colors[idx % colors.length];
        return {
          name: `${item.name} (Prediction)`,
          type: "line",
          showSymbol: false,
          data: item.data,
          color: color,
          lineStyle: {
            width: 2,
            color: color,
            type: "dashed"  // 虚线
          },
          itemStyle: {
            color: color
          },
          smooth: 0.4,
          connectNulls: true,
          z: 10  // Ensure prediction line is above original data
        };
      });
    
    if (predictionSeries.length > 0) {
      series.push(...predictionSeries);
      console.log(`Added ${predictionSeries.length} prediction series to chart`);
    }
  }

  // Get timestamp range of historical data
  const historicalTimestamps: number[] = [];
  seriesData.forEach(item => {
    item.data.forEach(point => {
      if (point[0] && !isNaN(point[0])) historicalTimestamps.push(point[0]);
    });
  });
  
  // Get timestamp range of prediction data (only use actual prediction data points, exclude historical data connection points)
  // Prediction data timestamps should all be after the last historical data time point
  const predictionTimestamps: number[] = [];
  const historicalMaxTime = historicalTimestamps.length > 0 ? Math.max(...historicalTimestamps) : 0;
  
  if (predictionSeriesData && predictionSeriesData.length > 0) {
    predictionSeriesData.forEach(item => {
      item.data.forEach(point => {
        if (point[0] && !isNaN(point[0])) {
          // Only collect timestamps greater than maximum historical data time (exclude historical data connection points)
          if (point[0] > historicalMaxTime) {
            predictionTimestamps.push(point[0]);
          }
        }
      });
    });
  }
  
  // Deduplicate and sort
  const uniquePredictionTimestamps = [...new Set(predictionTimestamps)].sort((a, b) => a - b);
  
  // Calculate minimum time (minimum value of historical data)
  const minTime = historicalTimestamps.length > 0 
    ? Math.min(...historicalTimestamps) 
    : (uniquePredictionTimestamps.length > 0 ? Math.min(...uniquePredictionTimestamps) : undefined);
  
  // Calculate maximum time
  // If there is prediction data, strictly limit to the last time point of prediction data (next 1 hour)
  // If there is no prediction data, use the maximum value of historical data
  let maxTime: number | undefined = undefined;
  if (uniquePredictionTimestamps.length > 0) {
    // Strictly use maximum timestamp of prediction data (exclude historical connection points)
    maxTime = Math.max(...uniquePredictionTimestamps);
    
    // Detailed debug information: print all prediction timestamps
    console.log('Time axis range calculation:', {
      historicalMaxTime: new Date(historicalMaxTime).toLocaleString('en-US'),
      historicalMaxTimeMs: historicalMaxTime,
      predictionTimestampsRaw: uniquePredictionTimestamps.map(ts => ({
        timestamp: ts,
        time: new Date(ts).toLocaleString('en-US')
      })),
      predictionCount: uniquePredictionTimestamps.length,
      minTime: new Date(minTime || 0).toLocaleString('zh-CN'),
      maxTime: new Date(maxTime).toLocaleString('zh-CN'),
      maxTimeMs: maxTime,
      predictionRange: {
        first: new Date(Math.min(...uniquePredictionTimestamps)).toLocaleString('zh-CN'),
        last: new Date(maxTime).toLocaleString('zh-CN')
      }
    });
  } else if (historicalTimestamps.length > 0) {
    maxTime = Math.max(...historicalTimestamps);
  }

  const allValues: number[] = [];
  const collectValues = (seriesList?: Array<{ data: Array<[number, number | null]> }>) => {
    if (!seriesList) return;
    seriesList.forEach(seriesItem => {
      seriesItem.data.forEach(point => {
        const value = point[1];
        if (typeof value === "number" && !isNaN(value)) {
          allValues.push(value);
        }
      });
    });
  };

  collectValues(seriesData);
  collectValues(predictionSeriesData);

  let yAxisMin: number | undefined;
  let yAxisMax: number | undefined;
  let axisDecimalPlaces: number | null = null;

  if (allValues.length > 0) {
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const range = maxVal - minVal;
    const paddingBase = range === 0 ? Math.max(Math.abs(maxVal) * 0.05, 0.05) : range * 0.1;
    const padding = paddingBase === 0 ? 0.05 : paddingBase;
    yAxisMin = minVal - padding;
    yAxisMax = maxVal + padding;

    const effectiveRange = (yAxisMax ?? maxVal) - (yAxisMin ?? minVal);
    if (effectiveRange < 0.1) {
      axisDecimalPlaces = 3;
    } else if (effectiveRange < 1) {
      axisDecimalPlaces = 2;
    } else if (effectiveRange < 10) {
      axisDecimalPlaces = 1;
    } else {
      axisDecimalPlaces = 0;
    }
  }

  const axisLabelFormatter = (value: number) => {
    if (!isFinite(value)) return "";
    if (axisDecimalPlaces === null) {
      return String(value);
    }
    return value.toFixed(axisDecimalPlaces);
  };

  const option: echarts.EChartsCoreOption = {
    backgroundColor: "transparent",
    animation: true,
    animationDuration: 750,
    animationEasing: "cubicOut",
    tooltip: {
      trigger: "axis",
      backgroundColor: "rgba(255, 255, 255, 0.95)",
      borderColor: colors[0],
      borderWidth: 2,
      textStyle: {
        color: "#1f2937",
        fontSize: 12
      },
      axisPointer: {
        type: "line",
        lineStyle: {
          color: colors[0],
          width: 2,
          type: "dashed"
        }
      },
      formatter: (params: any) => {
        if (!params || !Array.isArray(params)) return "";
        const timeValue = params[0].axisValue;
        let timeStr = "";
        if (typeof timeValue === "number") {
          const date = new Date(timeValue);
          if (!isNaN(date.getTime())) {
            const pad = (n: number) => String(n).padStart(2, "0");
            timeStr = `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
          }
        }
        let html = `<div style="margin-bottom: 6px; font-weight: 600;">${timeStr}</div>`;
        params.forEach((item: any) => {
          if (item.value !== null && item.value !== undefined) {
            const value = Array.isArray(item.value) ? item.value[item.value.length - 1] : item.value;
            const color = item.color || colors[0];
            html += `<div style="margin-top: 4px; display: flex; align-items: center;">
              <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background-color: ${color}; margin-right: 6px;"></span>
              <span>${item.seriesName || ""}: <strong style="color: ${color};">${value} ${unit}</strong></span>
            </div>`;
          }
        });
        return html;
      }
    },
    legend: {
      show: true,
      data: series.map(s => s.name),
      bottom: 10,
      left: "center",
      icon: "circle",
      itemWidth: 14,
      itemHeight: 14,
      itemGap: 20,
      textStyle: {
        fontSize: 12,
        padding: [0, 5, 0, 5]
      },
      backgroundColor: "rgba(255, 255, 255, 0.7)",
      borderRadius: 4,
      padding: [8, 12]
    },
    grid: {
      top: 50,
      right: 30,
      bottom: 100,
      left: 60,
      containLabel: true
    },
    xAxis: {
      type: "time",
      boundaryGap: false,
      min: minTime,
      max: maxTime,
      minInterval: 60 * 1000,
      maxInterval: 24 * 60 * 60 * 1000,
      axisLabel: {
        color: "#606266",
        fontSize: 12,
        rotate: -45,
        margin: 12,
        formatter: (value: number) => {
          if (!value || isNaN(value)) return "";
          const date = new Date(value);
          if (isNaN(date.getTime())) return "";
          const pad = (n: number) => String(n).padStart(2, "0");
          const month = date.getMonth() + 1;
          const day = date.getDate();
          const hours = date.getHours();
          const minutes = date.getMinutes();
          return `${month}/${day} ${pad(hours)}:${pad(minutes)}`;
        }
      },
      axisLine: {
        lineStyle: {
          color: "#E4E7ED",
          width: 1
        }
      },
      splitLine: {
        show: true,
        lineStyle: {
          color: "#EBEEF5",
          type: "dashed",
          width: 1
        }
      }
    },
    yAxis: {
      type: "value",
      name: unit,
      nameLocation: "end",
      nameTextStyle: {
        color: "#000000",
        fontSize: 12,
        padding: [0, 0, 0, 10]
      },
      axisLabel: {
        color: "#303133",
        fontSize: 12,
        formatter: axisLabelFormatter
      },
      axisTick: {
        lineStyle: {
          color: "#303133"
        }
      },
      axisLine: {
        lineStyle: {
          color: "#E4E7ED",
          width: 1
        }
      },
      splitLine: {
        show: true,
        lineStyle: {
          color: "#EBEEF5",
          type: "dashed",
          width: 1
        }
      },
      min: yAxisMin,
      max: yAxisMax,
      scale: true
    },
    series
  };

  return option;
}

// Process Crack Meter Data
function processCrackData(data: CrackDataPoint[], predictions?: PredictionPoint[]) {
  const seriesMap: Record<string, Array<[number, number | null]>> = {};
  
  CRACK_NUMBERS.forEach(num => {
    seriesMap[num] = [];
  });

  data.forEach(item => {
    if (item.number && CRACK_NUMBERS.includes(String(item.number))) {
      const timestamp = normalizeTimestamp(item.timestamp);
      const value = normalizeValue(item.data1);
      
      if (timestamp !== null) {
        seriesMap[String(item.number)].push([timestamp, value]);
      }
    }
  });

  // Sort by timestamp
  Object.keys(seriesMap).forEach(key => {
    seriesMap[key].sort((a, b) => a[0] - b[0]);
  });

  const series = Object.keys(seriesMap).map(name => ({
    name,
    data: seriesMap[name]
  }));

  // Process prediction data, ensure continuity with historical data
  // Backend has guaranteed prediction data is for next 1 hour (6 time steps), directly display all prediction data
  let predictionSeries: Array<{ name: string; data: Array<[number, number | null]> }> | undefined = undefined;
  if (predictions && predictions.length > 0) {
    predictionSeries = series.map((s) => {
      // Determine corresponding prediction field based on series name
      // According to training_data_assembly.md document:
      // crack_1: number 623622
      // crack_2: number 623641
      // crack_3: number 623628
      let valueKey: 'crack_1' | 'crack_2' | 'crack_3' | null = null;
      if (s.name === "623622") valueKey = 'crack_1';
      else if (s.name === "623641") valueKey = 'crack_2';
      else if (s.name === "623628") valueKey = 'crack_3';
      
      // Get the last point of historical data for this series
      const lastPoint = s.data.length > 0 ? s.data[s.data.length - 1] : null;
      
      // Build prediction data (display all prediction data, backend has guaranteed it's for next 1 hour)
      // Use time field instead of timestamp field, because time field is correct
      const predictionData = predictions
        .map(p => {
          // Use time field to parse timestamp
          let timestamp: number | null = null;
          if (p.time) {
            const date = new Date(p.time);
            if (!isNaN(date.getTime())) {
              timestamp = date.getTime(); // Convert to millisecond timestamp
            }
          }
          // Only use timestamp field as fallback if time field is invalid
          if (timestamp === null) {
            timestamp = normalizeTimestamp(p.timestamp);
          }
          const value = valueKey ? normalizeValue(p[valueKey]) : null;
          return timestamp !== null ? [timestamp, value] as [number, number | null] : null;
        })
        .filter((point): point is [number, number | null] => point !== null);
      
      // If historical data has a last point, add it to the beginning of prediction data to ensure continuity
      if (
        lastPoint &&
        lastPoint[0] &&
        lastPoint[1] !== null &&
        predictionData.length > 0
      ) {
        // Only append when historical data time is earlier than prediction start point, avoid two dashed lines at same time
        const firstPredTime = predictionData[0][0];
        if (lastPoint[0] < firstPredTime) {
          predictionData.unshift([lastPoint[0], lastPoint[1]]);
        }
      }
      
      // Debug information
      if (predictionData.length > 0) {
        console.log(`Prediction series ${s.name}: ${predictionData.length} data points`, {
          first: {
            timestamp: predictionData[0][0],
            time: new Date(predictionData[0][0]).toLocaleString('en-US'),
            value: predictionData[0][1]
          },
          last: {
            timestamp: predictionData[predictionData.length - 1][0],
            time: new Date(predictionData[predictionData.length - 1][0]).toLocaleString('en-US'),
            value: predictionData[predictionData.length - 1][1]
          },
          allTimestamps: predictionData.map(p => ({
            timestamp: p[0],
            time: new Date(p[0]).toLocaleString('en-US')
          })),
          valueKey,
          rawPredictions: predictions.map(p => {
            const timeFromTimeField = p.time ? new Date(p.time).getTime() : null;
            const timeFromTimestamp = normalizeTimestamp(p.timestamp);
            return {
              time: p.time,
              timeFromTimeField: timeFromTimeField ? new Date(timeFromTimeField).toLocaleString('en-US') : null,
              timestamp: p.timestamp,
              timestampMs: timeFromTimestamp,
              timeFromTimestamp: timeFromTimestamp ? new Date(timeFromTimestamp).toLocaleString('en-US') : null
            };
          })
        });
      }
      
      return {
        name: s.name,
        data: predictionData
      };
    });
    
    // Filter out prediction series without data
    predictionSeries = predictionSeries.filter(ps => ps.data && ps.data.length > 0);
    
    console.log(`Processed prediction series count: ${predictionSeries.length}`);
  } else {
    console.warn('No prediction data');
  }

  return { series, predictionSeries };
}

// Process Inclinometer Probe Data
function processTiltData(data: TiltDataPoint[], direction: "x" | "y") {
  const seriesMap: Record<string, Array<[number, number | null]>> = {};
  
  TILT_NUMBERS.forEach(num => {
    seriesMap[num] = [];
  });

  data.forEach(item => {
    if (item.number && TILT_NUMBERS.includes(String(item.number))) {
      const timestamp = normalizeTimestamp(item.timestamp);
      const value = direction === "x" 
        ? normalizeValue(item.data1)
        : normalizeValue(item.data2);
      
      if (timestamp !== null) {
        seriesMap[String(item.number)].push([timestamp, value]);
      }
    }
  });

  Object.keys(seriesMap).forEach(key => {
    seriesMap[key].sort((a, b) => a[0] - b[0]);
  });

  return Object.keys(seriesMap).map(name => ({
    name,
    data: seriesMap[name]
  }));
}

// Process Level Meter Data
function processLevelData(data: LevelDataPoint[]) {
  const seriesMap: Record<string, Array<[number, number | null]>> = {};
  
  LEVEL_NUMBERS.forEach(num => {
    seriesMap[num] = [];
  });

  data.forEach(item => {
    if (item.number && LEVEL_NUMBERS.includes(String(item.number))) {
      const timestamp = normalizeTimestamp(item.timestamp);
      const value = normalizeValue(item.data1);
      
      if (timestamp !== null) {
        seriesMap[String(item.number)].push([timestamp, value]);
      }
    }
  });

  Object.keys(seriesMap).forEach(key => {
    seriesMap[key].sort((a, b) => a[0] - b[0]);
  });

  return Object.keys(seriesMap).map(name => ({
    name,
    data: seriesMap[name]
  }));
}

// Process Water Level Meter Data
function processWaterLevelData(data: WaterLevelDataPoint[]) {
  const seriesMap: Record<string, Array<[number, number | null]>> = {};
  
  // Only display device with number=478967
  const TARGET_NUMBER = "478967";
  
  // Only keep data from target device
  const filteredData = data.filter(item => {
    const itemNumber = String(item.number || "").replace(/\D/g, "").padStart(6, "0");
    return itemNumber === TARGET_NUMBER;
  });

  if (filteredData.length > 0) {
    seriesMap[TARGET_NUMBER] = [];
    
    filteredData.forEach(item => {
      const timestamp = normalizeTimestamp(item.timestamp);
      const value = normalizeValue(item.data1);
      
      if (timestamp !== null) {
        seriesMap[TARGET_NUMBER].push([timestamp, value]);
      }
    });

    seriesMap[TARGET_NUMBER].sort((a, b) => a[0] - b[0]);
  }

  return Object.keys(seriesMap).map(name => ({
    name,
    data: seriesMap[name]
  }));
}

// Update Charts
async function updateCharts() {
  try {
    // Fetch all data (including prediction data)
    const [crackData, tiltData, levelData, waterLevelData, predictionData] = await Promise.all([
      fetchCrackData(),
      fetchTiltData(),
      fetchLevelData(),
      fetchWaterLevelData(),
      fetchPredictionData()
    ]);

    // Update crack meter chart
    if (crackChart && crackChartRef.value) {
      const { series: crackSeries, predictionSeries } = processCrackData(crackData, predictionData);
      const crackOption = createChartOption(crackSeries, "mm", predictionSeries);
      crackChart.setOption(crackOption, true);
    }

    // Update inclinometer probe-X chart
    if (tiltXChart && tiltXChartRef.value) {
      const tiltXSeries = processTiltData(tiltData, "x");
      const tiltXOption = createChartOption(tiltXSeries, "mm");
      tiltXChart.setOption(tiltXOption, true);
    }

    // Update inclinometer probe-Y chart
    if (tiltYChart && tiltYChartRef.value) {
      const tiltYSeries = processTiltData(tiltData, "y");
      const tiltYOption = createChartOption(tiltYSeries, "mm");
      tiltYChart.setOption(tiltYOption, true);
    }

    // Update level meter chart
    if (levelChart && levelChartRef.value) {
      const levelSeries = processLevelData(levelData);
      const levelOption = createChartOption(levelSeries, "mm");
      levelChart.setOption(levelOption, true);
    }

    // Update water level meter chart
    if (waterLevelChart && waterLevelChartRef.value) {
      const waterLevelSeries = processWaterLevelData(waterLevelData);
      const waterLevelOption = createChartOption(waterLevelSeries, "mm");
      waterLevelChart.setOption(waterLevelOption, true);
    }

    // Update last update time
    lastUpdateTime.value = new Date().toLocaleString("en-US");
  } catch (error) {
    console.error("Failed to fetch data:", error);
  }
}

// Initialize Charts
function initCharts() {
  if (crackChartRef.value) {
    crackChart = echarts.init(crackChartRef.value);
  }
  if (tiltXChartRef.value) {
    tiltXChart = echarts.init(tiltXChartRef.value);
  }
  if (tiltYChartRef.value) {
    tiltYChart = echarts.init(tiltYChartRef.value);
  }
  if (levelChartRef.value) {
    levelChart = echarts.init(levelChartRef.value);
  }
  if (waterLevelChartRef.value) {
    waterLevelChart = echarts.init(waterLevelChartRef.value);
  }

  // Listen for window resize events
  window.addEventListener("resize", handleResize);
}

// Handle window resize
function handleResize() {
  crackChart?.resize();
  tiltXChart?.resize();
  tiltYChart?.resize();
  levelChart?.resize();
  waterLevelChart?.resize();
}

// Manual data refresh
function refreshData() {
  updateCharts();
}

// Start automatic refresh
function startAutoRefresh() {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
  refreshTimer = window.setInterval(() => {
    updateCharts();
  }, REFRESH_INTERVAL);
}

// Component mounted
onMounted(() => {
  initCharts();
  updateCharts();
  startAutoRefresh();
});

// Component unmounted
onUnmounted(() => {
  if (refreshTimer) {
    clearInterval(refreshTimer);
  }
  window.removeEventListener("resize", handleResize);
  crackChart?.dispose();
  tiltXChart?.dispose();
  tiltYChart?.dispose();
  levelChart?.dispose();
  waterLevelChart?.dispose();
});
</script>

<style scoped>
.shm-monitor-container {
  padding: 20px;
  background-color: #f5f7fa;
  min-height: 100vh;
}

.image-container {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
}

.image-content-wrapper {
  display: flex;
  gap: 20px;
  align-items: flex-start;
}

.image-wrapper {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
}

.sensor-layout-image {
  max-width: 100%;
  height: auto;
  border-radius: 8px;
}

.header-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header h2 {
  margin: 0;
  font-size: 20px;
  color: #303133;
}

.refresh-info {
  display: flex;
  align-items: center;
  gap: 15px;
}

.refresh-info span {
  color: #606266;
  font-size: 14px;
}

.building-intro {
  margin-top: 0;
}

.building-intro h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  border-left: 4px solid #409eff;
  padding-left: 10px;
}

.building-intro h3:not(:first-child) {
  margin-top: 24px;
}

.building-intro p {
  margin: 0;
  line-height: 1.8;
  color: #606266;
  font-size: 14px;
  text-align: justify;
}

.chart-container {
  background-color: #fff;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  margin-bottom: 20px;
}

.chart-container.full-width {
  width: 100%;
}

.chart-container.half-width {
  flex: 1;
  margin: 0 10px;
}

.chart-row {
  display: flex;
  gap: 20px;
}

.chart-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 2px solid #409eff;
}

.chart {
  width: 100%;
  height: 400px;
}

/* 响应式设计 */
@media (max-width: 1200px) {
  .image-content-wrapper {
    flex-direction: column;
  }
  
  .image-wrapper {
    width: 100%;
  }
  
  .chart-row {
    flex-direction: column;
  }
  
  .chart-container.half-width {
    margin: 0;
    margin-bottom: 20px;
  }
}
</style>

