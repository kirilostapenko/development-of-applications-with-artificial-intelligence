"""Запуск із pr5: python -m compare.run. Локальний навчальний експеримент."""
import json
import time
from pathlib import Path
from app import documents, embeddings, index, keyword


def main():
    root = Path(__file__).parent
    queries = json.loads((root/'queries.json').read_text(encoding='utf-8'))['запити']
    report = ['# Результати порівняння', '', 'Час семантичного пошуку включає кодування запиту. Модель прогріта. Експеримент локальний, без обмежень клієнтського веб-API.', '']
    embeddings.get_model()
    original = documents.CHUNK_SIZE
    for size in (original, original * 2):
        documents.CHUNK_SIZE = size
        chunks = documents.load_chunks()
        idx = index.build(chunks, embeddings.embed_passages([c.text for c in chunks]), embeddings.MODEL_NAME)
        kw = keyword.build(chunks)
        report += [f'## Розмір {size} символів ({len(chunks)} фрагментів)', '', '| Запит | Метод | Фільтр | Позиція | Перший результат / оцінка | Очікувана оцінка | Вище очікуваного | Час, мс |', '|---|---|---|---|---|---|---|---|']
        counts = {m:[0,0,0] for m in ('semantic','keyword')}
        positives, negatives = [], []
        for q in queries:
            text, expected = q['запит'], q['очікуваний_документ']
            variants = [{}]
            if q.get('фільтр'): variants.append(q['фільтр'])
            if q['вид']=='чинне проти застарілого': variants.append({'status':'чинний'})
            if q['вид']=='внутрішній документ': variants.append({'audience':'клієнти'})
            for filters in variants:
                for method in counts:
                    t = time.perf_counter()
                    hits = index.search(idx, embeddings.embed_query(text), 5, filters, None) if method=='semantic' else keyword.search(kw,text,5,filters)
                    ms = (time.perf_counter()-t)*1000
                    rank = next((i+1 for i,h in enumerate(hits) if h.chunk.source==expected),None)
                    first = f'{hits[0].chunk.source} / {hits[0].score:.4f}' if hits else '—'
                    above = ', '.join(f'{h.chunk.source} ({h.score:.4f})' for h in hits[:rank-1 if rank else len(hits)]) or '—'
                    score = f'{hits[rank-1].score:.4f}' if rank else '—'
                    report.append(f'| {text} | {method} | {filters or "—"} | {rank or "—"} | {first} | {score} | {above} | {ms:.2f} |')
                    if not filters and expected:
                        c=counts[method]; c[0]+=rank==1; c[1]+=rank is not None and rank<=3; c[2]+=1
                    if not filters and method=='semantic':
                        if expected: positives.append(hits[rank-1].score if rank else None)
                        elif hits: negatives.append(hits[0].score)
        report.append('')
        for m,(a,b,n) in counts.items(): report.append(f'- {m}: hit@1 = {a}/{n} ({a/n:.1%}); hit@3 = {b}/{n} ({b/n:.1%}).')
        if negatives:
            threshold = max(negatives)+0.0001
            if threshold<=1:
                report.append(f'- Експериментальний поріг {threshold:.4f}: відсікає всі {len(negatives)} позаколекційні запити цього набору; втрачає {sum(x is not None and x<threshold for x in positives)} із {sum(x is not None for x in positives)} знайдених очікуваних документів. Потрібна перевірка на нових запитах.')
        report.append('')
    (root/'results.md').write_text('\n'.join(report),encoding='utf-8')
    print('Готово: compare/results.md')

if __name__=='__main__': main()
