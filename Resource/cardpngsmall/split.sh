#!/bin/bash

# 파일 카운터 초기화
counter=0

# 현재 디렉토리의 JPEG 이미지를 대상으로 반복
for file in *.png; do
  # 폴더 이름을 파일 카운터로 설정 (printf로 패딩 추가)
  dirname=$(printf "%d" $counter)
  
  # 해당 이름의 디렉토리 생성
  mkdir -p "$dirname"
  
  # 파일을 새 디렉토리로 이동 (파일 이름도 동일하게 설정)
  mv "$file" "$dirname/$dirname.jpg"
  
  # 파일 카운터 증가
  counter=$((counter+1))
done

