#!/bin/bash

# 새로운 종합 디렉토리 생성
mkdir -p combined_directory

# 0부터 47까지의 디렉토리 번호를 반복
for i in {0..47}; do
  # 각 번호에 대해 새 디렉토리 생성
  mkdir -p combined_directory/$i

  # 각 원래 디렉토리에서 해당 번호 디렉토리의 내용을 새 디렉토리로 이동
  for dir in cardpng cardpngdark cardpnglight cardpngsmall cardpngsmalldark cardpngsmallgrey cardpngsmalllight; do
    if [ -d "$dir/$i" ]; then  # 해당 번호의 디렉토리가 존재하는 경우에만
      # 내용을 새 위치로 이동하면서 파일명 변경
      for file in "$dir/$i/"*; do
        # 파일명에서 확장자를 분리
        filename=$(basename -- "$file")
        extension="${filename##*.}"
        filename="${filename%.*}"
        
        # 새 파일명 생성: 원래 디렉토리 이름 추가
        new_filename="${filename}_${dir}.${extension}"
        
        # 파일 이동
        cp "$file" "combined_directory/$i/$new_filename"
      done
    fi
  done
done

