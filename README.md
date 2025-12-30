# rcat-voice


### 🏗️ 技术架构

#### **核心模块** （7个文件）

1. **deepseek.rs** - LLM 流式客户端

* 使用

  ```
  async-openai
  ```

  库
* 支持 DeepSeek/OpenAI 兼容 API
* SSE 流式输出（逐字返回）

1. **sentence_splitter.rs** - 句子边界检测 ⭐

* 纯 Rust 实现（

  ```
  unicode-segmentation
  ```

  ）
* 中英文语言检测
* 智能断句规则（支持缩写词如 Dr./Mr.）
* 完整测试覆盖

1. **chunker.rs** - 自适应文本分块器

* **动态调整策略** ：根据播放队列长度调整分块大小
  * 首段：10-20字符（快速首播）
  * 低缓冲区：20-45字符（快速补充）
  * 高缓冲区：80-140字符（批量处理）
* 智能边界检测（强边界：。！？，弱边界：，；：）
* 估算播放时长（180ms/字符）

1. **player.rs** - 音频播放器

* 异步播放管理
* **详细性能指标输出** ：
  * LLM首字时延
  * 分段器延迟
  * 首播时延
  * 音频播放完成时间
* 取消/停止支持

1. **tts.rs** - TTS 引擎抽象层

* 跨平台系统 TTS：

  * **Windows** : PowerShell + System.Speech
  * **macOS** :

  ```
  say
  ```

  命令

  * **Linux** :

  ```
  spd-say
  ```
* 异步进程管理
* 支持中断/停止

1. **main.rs** - 管道编排

* 三阶段管道：LLM → Chunker → Player
* 基于 tokio channel 的流式通信
* 共享状态管理（队列估算）

1. **lib.rs** - 库入口

---

### 📈 最近开发重点（根据会话历史）

#### **2025-12-30 会话：增强 TTS 指标**

核心目标：实现**全链路性能追踪**

 **实现的关键指标** ：

* **t0** : Task Start（任务开始）
* **t1** : TTFT（Time To First Token） - LLM首字时延
* **t2** : TTST（Time To Send Text）- 分段器延迟
* **TTFA** （Time To First Audio）- 首播时延
* **TTPT** （Time To Play Time）- 播放完成时间
* **TTS_TIME** - 总处理时间

 **技术选型** ：

* ✅ 采用纯 Rust 实现（

  ```
  unicode-segmentation
  ```

  ）
* ❌ 放弃 Python 库（如

  ```
  nltk
  ```

  ）- 兼容性问题

---

### 📦 依赖栈

<pre><div><div class="min-h-7 relative box-border flex flex-row items-center justify-between rounded-t border border-b-0 border-gray-500/25 px-2 py-0.5"><div class="font-sans text-sm text-ide-text-color opacity-60">toml</div><div><div class="flex flex-row items-center gap-0.5"><div class="rounded-sm p-1 cursor-pointer opacity-60 hover:bg-gray-500/25 hover:opacity-100"><span data-tooltip-id="At mention" class="text-ide-text-color"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true" data-slot="icon" class="h-3.5 w-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M16.5 12a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0Zm0 0c0 1.657 1.007 3 2.25 3S21 13.657 21 12a9 9 0 1 0-2.636 6.364M16.5 12V8.25"></path></svg></span></div><div class="rounded-sm p-1 cursor-pointer opacity-60 hover:bg-gray-500/25 hover:opacity-100"><span data-tooltip-id="Copy" class="text-ide-text-color"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-copy h-3.5 w-3.5"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg></span></div></div></div></div><div class="language-toml relative overflow-hidden rounded-b border-x border-b border-gray-500/25 bg-ide-editor-background p-2" aria-label="highlighted-code-language-toml"><div class="w-full h-full text-xs cursor-text"><div class="code-block"><div class="code-line" data-line-number="1" data-line-start="1" data-line-end="1"><div class="line-content"><span class="mtk1">tokio (异步运行时)</span></div></div><div class="code-line" data-line-number="2" data-line-start="2" data-line-end="2"><div class="line-content"><span class="mtk1">async-openai (LLM 客户端)</span></div></div><div class="code-line" data-line-number="3" data-line-start="3" data-line-end="3"><div class="line-content"><span class="mtk1">unicode-segmentation (句子分割)</span></div></div><div class="code-line" data-line-number="4" data-line-start="4" data-line-end="4"><div class="line-content"><span class="mtk1">reqwest (HTTP 客户端)</span></div></div><div class="code-line" data-line-number="5" data-line-start="5" data-line-end="5"><div class="line-content"><span class="mtk1">serde_json (数据序列化)</span></div></div><div class="code-line" data-line-number="6" data-line-start="6" data-line-end="6"><div class="line-content"><span class="mtk1">tracing (日志追踪)</span></div></div><div class="code-line" data-line-number="7" data-line-start="7" data-line-end="7"><div class="line-content"><span class="mtk1">anyhow (错误处理)</span></div></div></div></div></div></div></pre>

---

### ✅ 项目完成度

 **已实现功能** ：

* ✅ 完整的流式管道（LLM → TTS）
* ✅ 自适应文本分块
* ✅ 中英文句子分割
* ✅ 详细性能指标
* ✅ 跨平台 TTS 支持
* ✅ 取消/中断机制
* ✅ 单元测试（

  ```
  sentence_splitter
  ```

  ）

 **当前状态** ：

* 📁 光标位置：

  sentence_splitter.rs 第54行（断句逻辑核心）
* 🔧 最近修改：性能指标追踪、句子分割器
* 📊 项目成熟度： **POC 阶段** （可运行的原型）

---

### 🚀 优化空间/潜在改进

1. **句子分割器** （当前打开文件）

* 可扩展更多语言支持
* 更复杂的缩写词处理

1. **TTS 引擎**
   * 可集成远程 TTS 服务（更高质量）
   * 语音参数配置（语速/音调）
2. **测试覆盖**
   * 需要为其他模块添加测试
   * 集成测试
3. **文档**
   * 无 README（建议添加）
   * 缺少使用说明
4. **错误处理**
   * 可改进错误恢复机制

---

### 🎯 项目定位总结

这是一个 **高度优化的实时流式 TTS 系统** ，重点在于：

* 最小化延迟（特别是首播延迟 TTFA）
* 智能资源管理（自适应分块）
* 详细的性能观测

从代码质量看，这是一个**结构清晰、设计合理**的 Rust 异步项目，适合作为流式 AI 应用的基础框架。

### 🏗️ 技术架构

#### **核心模块** （7个文件）

1. **deepseek.rs** - LLM 流式客户端

* 使用

  ```
  async-openai
  ```

  库
* 支持 DeepSeek/OpenAI 兼容 API
* SSE 流式输出（逐字返回）

1. **sentence_splitter.rs** - 句子边界检测 ⭐

* 纯 Rust 实现（

  ```
  unicode-segmentation
  ```

  ）
* 中英文语言检测
* 智能断句规则（支持缩写词如 Dr./Mr.）
* 完整测试覆盖

1. **chunker.rs** - 自适应文本分块器

* **动态调整策略** ：根据播放队列长度调整分块大小
  * 首段：10-20字符（快速首播）
  * 低缓冲区：20-45字符（快速补充）
  * 高缓冲区：80-140字符（批量处理）
* 智能边界检测（强边界：。！？，弱边界：，；：）
* 估算播放时长（180ms/字符）

1. **player.rs** - 音频播放器

* 异步播放管理
* **详细性能指标输出** ：
  * LLM首字时延
  * 分段器延迟
  * 首播时延
  * 音频播放完成时间
* 取消/停止支持

1. **tts.rs** - TTS 引擎抽象层

* 跨平台系统 TTS：

  * **Windows** : PowerShell + System.Speech
  * **macOS** :

  ```
  say
  ```

  命令

  * **Linux** :

  ```
  spd-say
  ```
* 异步进程管理
* 支持中断/停止

1. **main.rs** - 管道编排

* 三阶段管道：LLM → Chunker → Player
* 基于 tokio channel 的流式通信
* 共享状态管理（队列估算）

1. **lib.rs** - 库入口

---

### 📈 最近开发重点（根据会话历史）

#### **2025-12-30 会话：增强 TTS 指标**

核心目标：实现**全链路性能追踪**

 **实现的关键指标** ：

* **t0** : Task Start（任务开始）
* **t1** : TTFT（Time To First Token） - LLM首字时延
* **t2** : TTST（Time To Send Text）- 分段器延迟
* **TTFA** （Time To First Audio）- 首播时延
* **TTPT** （Time To Play Time）- 播放完成时间
* **TTS_TIME** - 总处理时间

 **技术选型** ：

* ✅ 采用纯 Rust 实现（

  ```
  unicode-segmentation
  ```

  ）
* ❌ 放弃 Python 库（如

  ```
  nltk
  ```

  ）- 兼容性问题

---

### 📦 依赖栈

<pre><div><div class="min-h-7 relative box-border flex flex-row items-center justify-between rounded-t border border-b-0 border-gray-500/25 px-2 py-0.5"><div class="font-sans text-sm text-ide-text-color opacity-60">toml</div><div><div class="flex flex-row items-center gap-0.5"><div class="rounded-sm p-1 cursor-pointer opacity-60 hover:bg-gray-500/25 hover:opacity-100"><span data-tooltip-id="At mention" class="text-ide-text-color"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" aria-hidden="true" data-slot="icon" class="h-3.5 w-3.5"><path stroke-linecap="round" stroke-linejoin="round" d="M16.5 12a4.5 4.5 0 1 1-9 0 4.5 4.5 0 0 1 9 0Zm0 0c0 1.657 1.007 3 2.25 3S21 13.657 21 12a9 9 0 1 0-2.636 6.364M16.5 12V8.25"></path></svg></span></div><div class="rounded-sm p-1 cursor-pointer opacity-60 hover:bg-gray-500/25 hover:opacity-100"><span data-tooltip-id="Copy" class="text-ide-text-color"><svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-copy h-3.5 w-3.5"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg></span></div></div></div></div><div class="language-toml relative overflow-hidden rounded-b border-x border-b border-gray-500/25 bg-ide-editor-background p-2" aria-label="highlighted-code-language-toml"><div class="w-full h-full text-xs cursor-text"><div class="code-block"><div class="code-line" data-line-number="1" data-line-start="1" data-line-end="1"><div class="line-content"><span class="mtk1">tokio (异步运行时)</span></div></div><div class="code-line" data-line-number="2" data-line-start="2" data-line-end="2"><div class="line-content"><span class="mtk1">async-openai (LLM 客户端)</span></div></div><div class="code-line" data-line-number="3" data-line-start="3" data-line-end="3"><div class="line-content"><span class="mtk1">unicode-segmentation (句子分割)</span></div></div><div class="code-line" data-line-number="4" data-line-start="4" data-line-end="4"><div class="line-content"><span class="mtk1">reqwest (HTTP 客户端)</span></div></div><div class="code-line" data-line-number="5" data-line-start="5" data-line-end="5"><div class="line-content"><span class="mtk1">serde_json (数据序列化)</span></div></div><div class="code-line" data-line-number="6" data-line-start="6" data-line-end="6"><div class="line-content"><span class="mtk1">tracing (日志追踪)</span></div></div><div class="code-line" data-line-number="7" data-line-start="7" data-line-end="7"><div class="line-content"><span class="mtk1">anyhow (错误处理)</span></div></div></div></div></div></div></pre>

---

### ✅ 项目完成度

 **已实现功能** ：

* ✅ 完整的流式管道（LLM → TTS）
* ✅ 自适应文本分块
* ✅ 中英文句子分割
* ✅ 详细性能指标
* ✅ 跨平台 TTS 支持
* ✅ 取消/中断机制
* ✅ 单元测试（

  ```
  sentence_splitter
  ```

  ）

 **当前状态** ：

* 📁 光标位置：

  sentence_splitter.rs 第54行（断句逻辑核心）
* 🔧 最近修改：性能指标追踪、句子分割器
* 📊 项目成熟度： **POC 阶段** （可运行的原型）

---

### 🚀 优化空间/潜在改进

1. **句子分割器** （当前打开文件）

* 可扩展更多语言支持
* 更复杂的缩写词处理

1. **TTS 引擎**
   * 可集成远程 TTS 服务（更高质量）
   * 语音参数配置（语速/音调）
2. **测试覆盖**
   * 需要为其他模块添加测试
   * 集成测试
3. **文档**
   * 无 README（建议添加）
   * 缺少使用说明
4. **错误处理**
   * 可改进错误恢复机制

---

### 🎯 项目定位总结

这是一个 **高度优化的实时流式 TTS 系统** ，重点在于：

* 最小化延迟（特别是首播延迟 TTFA）
* 智能资源管理（自适应分块）
* 详细的性能观测

从代码质量看，这是一个**结构清晰、设计合理**的 Rust 异步项目，适合作为流式 AI 应用的基础框架。
