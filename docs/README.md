# rcat-voice 文档索引

本目录包含 rcat-voice 项目的技术文档。

## 文档列表

| 文档 | 用途 | 适合读者 |
|------|------|---------|
| [ARCHITECTURE.md](./ARCHITECTURE.md) | 系统架构、数据流图、模块依赖 | 开发者、贡献者 |
| [FEATURE_MAP.md](./FEATURE_MAP.md) | 功能-代码映射，自然语言定位代码 | AI 辅助开发、新贡献者 |
| [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) | 常见问题、故障排查、调试技巧 | 集成者、运维 |
| [optimized.md](./optimized.md) | 未实现/待优化事项清单 | 开发者、维护者 |

## 阅读建议

- **新手入门**: 先阅读 [../README.md](../README.md) 了解快速开始，再看 ARCHITECTURE.md
- **贡献代码**: 阅读 FEATURE_MAP.md 快速定位功能对应的代码位置
- **遇到问题**: 先查 TROUBLESHOOTING.md，再搜索 GitHub Issues

## 文档维护

更新代码时请同步更新相关文档，特别是：
- 新增/修改环境变量 → 更新 README.md 和 FEATURE_MAP.md
- 新增功能模块 → 更新 ARCHITECTURE.md 和 FEATURE_MAP.md
- 修复问题 → 更新 TROUBLESHOOTING.md
