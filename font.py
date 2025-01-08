import matplotlib.font_manager

# 列出所有可用的字體
fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
chinese_fonts = [f for f in fonts if any(s in f for s in ['Hei', 'Ming', 'Song', '黑體', '明朝', 'Gothic'])]
print("可用的中文字體：")
for font in chinese_fonts:
    print(font)